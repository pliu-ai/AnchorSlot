"""UniSeg and CoNeMOS mechanisms on the common nnU-Net v2 PlainConvUNet.

Sources: yeerwen/UniSeg e3ae40d (Prompt_generic_UNet.py), BBillot/CoNeMOS
c440794 (unet.py, film.py). Independently implemented PyTorch adaptations.
"""
import torch
from torch import nn
from torch.nn import functional as F
from dynamic_network_architectures.building_blocks.simple_conv_blocks import ConvDropoutNormReLU

TASKS = ((4, 5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
         (19, 20), (25, 26), (21, 22), (23, 24), (28, 29), (30, 31),
         (17, 18), (1, 2, 3, 27))
assert sorted(c for task in TASKS for c in task) == list(range(1, 32))


class UniSegCellMap(nn.Module):
    def __init__(self, backbone, prompt_shape=(5, 5, 5)):
        super().__init__()
        self.backbone = backbone
        channels = backbone.encoder.output_channels[-1]
        self.universal_prompt = nn.Parameter(torch.randn(1, len(TASKS), *prompt_shape))
        middle = (channels + len(TASKS)) // 4
        layers = []
        for a, b in ((channels + len(TASKS), middle), (middle, middle), (middle, len(TASKS))):
            layers.extend([nn.Conv3d(a, b, 3, padding=1),
                           nn.InstanceNorm3d(b, eps=1e-5, affine=True),
                           nn.LeakyReLU(1e-2, inplace=True)])
        self.fuse = nn.Sequential(*layers)
        up = self.decoder.transpconvs[0]
        self.decoder.transpconvs[0] = nn.ConvTranspose3d(
            channels + 1, up.out_channels, up.kernel_size, up.stride,
            up.padding, up.output_padding, up.groups, up.bias is not None, up.dilation)

    @property
    def decoder(self):
        return self.backbone.decoder

    def encode(self, x):
        skips = self.backbone.encoder(x)
        prompt = self.universal_prompt
        if tuple(prompt.shape[2:]) != tuple(skips[-1].shape[2:]):
            prompt = F.interpolate(prompt, skips[-1].shape[2:], mode='trilinear', align_corners=False)
        dynamic = self.fuse(torch.cat((skips[-1], prompt.expand(x.shape[0], -1, -1, -1, -1)), 1))
        return skips, dynamic

    def decode(self, encoded, query):
        skips, dynamic = encoded
        query = query.to(dynamic.device).long().reshape(-1)
        prompt = dynamic[torch.arange(len(query), device=query.device), query][:, None]
        local_skips = [*skips[:-1], torch.cat((skips[-1], prompt), 1)]
        return self.decoder(local_skips)

    def forward(self, x, query):
        return self.decode(self.encode(x), query)


class FiLM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.affine = nn.Linear(64, channels * 2)
        nn.init.xavier_uniform_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x, condition):
        gamma, beta = self.affine(condition).chunk(2, 1)
        shape = (x.shape[0], x.shape[1], *([1] * (x.ndim - 2)))
        return (1 + gamma.reshape(shape)) * x + beta.reshape(shape)


class CoNeMOSCellMap(nn.Module):
    """Full convolution-block FiLM, four-layer one-hot MLP, binary logit.

nnU-Net conv -> InstanceNorm -> LeakyReLU -> FiLM is the declared port;
upstream uses conv + activation -> optional normalization -> FiLM.
Transpose convolutions retain the common backbone's implementation.
"""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        layers = []
        for i in range(4):
            layers.extend([nn.Linear(31 if i == 0 else 64, 64), nn.LeakyReLU(0.2)])
        self.condition_mlp = nn.Sequential(*layers)
        # Walk unique blocks; the decoder also references the encoder.
        blocks = [m for m in backbone.modules() if isinstance(m, ConvDropoutNormReLU)]
        self.feature_films = nn.ModuleList([FiLM(m.output_channels) for m in blocks])
        self._condition = None
        for block, film in zip(blocks, self.feature_films):
            block.register_forward_hook(self._hook(film))
        self.output_films = nn.ModuleList([FiLM(1) for _ in self.decoder.seg_layers])
        for head, film in zip(self.decoder.seg_layers, self.output_films):
            head.register_forward_hook(self._hook(film))

    @property
    def decoder(self):
        return self.backbone.decoder

    def _hook(self, film):
        def apply_film(module, inputs, output):
            if self._condition is None:
                raise RuntimeError('CoNeMOS requires an explicit label query')
            return film(output, self._condition)
        return apply_film

    def forward(self, x, query):
        onehot = F.one_hot(query.long(), 31).to(dtype=x.dtype, device=x.device)
        self._condition = self.condition_mlp(onehot)
        try:
            return self.backbone(x)
        finally:
            self._condition = None


class QueryView(nn.Module):
    """Inference view: predict one task, then fuse only after spatial aggregation."""
    def __init__(self, network, query):
        super().__init__()
        self.network = network
        self.query = int(query)

    def forward(self, x):
        return self.network(x, torch.full((x.shape[0],), self.query, device=x.device, dtype=torch.long))
