from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn

from .network_hierarchical_parallel_anchorslot import HierarchicalParallelAnchorSlotUNet
from .resolution_adaptive_mapping import NUM_PARENT_CLASSES


class ResolutionAdaptiveHierarchicalAnchorSlotUNet(HierarchicalParallelAnchorSlotUNet):
    """Hierarchical AnchorSlot with physical-scale conditioning and auxiliary heads.

    Resolution FiLM is applied at every decoder stage. The 17-channel parent
    head predicts the official overlapping CellMap DAG nodes directly, while a
    compact class-agnostic separation head predicts boundary and axis affinities.
    """

    def __init__(
        self,
        backbone: nn.Module,
        code_dim: int = 64,
        reference_voxel_size: Sequence[float] = (4.0, 4.0, 4.0),
    ) -> None:
        super().__init__(backbone=backbone, code_dim=code_dim)
        decoder_stage_channels = [int(layer.in_channels) for layer in self.decoder.seg_layers]
        conv_type = type(self.decoder.seg_layers[0])
        spatial_dims = 3 if conv_type is nn.Conv3d else 2
        self.num_separation_channels = spatial_dims + 1
        reference = torch.as_tensor(reference_voxel_size, dtype=torch.float32)
        if reference.numel() != spatial_dims:
            if reference.numel() == 3 and spatial_dims == 2:
                reference = reference[-2:]
            else:
                raise ValueError(
                    f"reference_voxel_size must have {spatial_dims} values, got {tuple(reference_voxel_size)}"
                )
        self.register_buffer("reference_voxel_size", reference, persistent=True)

        self.resolution_encoder = nn.Sequential(
            nn.Linear(spatial_dims, code_dim),
            nn.SiLU(inplace=True),
            nn.Linear(code_dim, code_dim),
        )
        self.resolution_to_pair = nn.Linear(code_dim, code_dim, bias=False)
        self.resolution_film = nn.ModuleList(
            [nn.Linear(code_dim, channels * 2) for channels in decoder_stage_channels]
        )
        self.parent_heads = nn.ModuleList(
            [conv_type(channels, NUM_PARENT_CLASSES, kernel_size=1, bias=True) for channels in decoder_stage_channels]
        )
        self.separation_heads = nn.ModuleList(
            [conv_type(channels, self.num_separation_channels, kernel_size=1, bias=True) for channels in decoder_stage_channels]
        )
        nn.init.zeros_(self.resolution_to_pair.weight)
        for film in self.resolution_film:
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)

    def _resolution_embedding(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        voxel_size: torch.Tensor | Sequence[float] | None,
    ) -> torch.Tensor:
        if voxel_size is None:
            values = self.reference_voxel_size[None].expand(batch_size, -1)
        else:
            values = torch.as_tensor(voxel_size, device=device, dtype=torch.float32)
            if values.ndim == 1:
                values = values[None].expand(batch_size, -1)
            if values.shape != (batch_size, self.reference_voxel_size.numel()):
                raise ValueError(
                    f"voxel_size must have shape [{batch_size}, {self.reference_voxel_size.numel()}], "
                    f"got {tuple(values.shape)}"
                )
        normalized = torch.log2(values / self.reference_voxel_size[None].to(device)).clamp(-4.0, 4.0)
        return self.resolution_encoder(normalized.to(dtype=torch.float32)).to(dtype=dtype)

    def _condition_features(
        self,
        features: torch.Tensor,
        resolution_embedding: torch.Tensor,
        stage_index: int,
    ) -> torch.Tensor:
        gamma, beta = self.resolution_film[stage_index](resolution_embedding).chunk(2, dim=1)
        broadcast = (features.shape[0], features.shape[1], *([1] * (features.ndim - 2)))
        gamma = 0.1 * torch.tanh(gamma).view(broadcast)
        beta = beta.view(broadcast)
        return features * (1.0 + gamma) + beta

    def decode_from_skips(
        self,
        skips: List[torch.Tensor],
        return_hierarchy: bool = False,
        voxel_size: torch.Tensor | Sequence[float] | None = None,
    ):
        lres_input = skips[-1]
        semantic_outputs = []
        coarse_outputs = []
        slot_outputs = []
        parent_outputs = []
        separation_outputs = []
        resolution_embedding = self._resolution_embedding(
            skips[0].shape[0], skips[0].device, skips[0].dtype, voxel_size
        )
        base_pairs = self.group_embeddings[None, :, None, :] + self.anchor_slot_embeddings[None, None, :, :]
        pair_codes = self.pair_encoder(
            base_pairs + self.resolution_to_pair(resolution_embedding)[:, None, None, :]
        )

        for stage_idx in range(len(self.decoder.stages)):
            x = self.decoder.transpconvs[stage_idx](lres_input)
            x = torch.cat((x, skips[-(stage_idx + 2)]), dim=1)
            x = self.decoder.stages[stage_idx](x)
            x = self._condition_features(x, resolution_embedding, stage_idx)
            emit = self.decoder.deep_supervision or stage_idx == len(self.decoder.stages) - 1
            if emit:
                coarse_logits = self.decoder.seg_layers[stage_idx](x)
                slot_logits = self.slot_heads[stage_idx](x, pair_codes)
                semantic_outputs.append(self.compose_semantic_log_probs(coarse_logits, slot_logits))
                coarse_outputs.append(coarse_logits)
                slot_outputs.append(slot_logits)
                parent_outputs.append(self.parent_heads[stage_idx](x))
                separation_outputs.append(self.separation_heads[stage_idx](x))
            lres_input = x

        outputs = {
            "semantic_logits": list(reversed(semantic_outputs)),
            "coarse_logits": list(reversed(coarse_outputs)),
            "slot_logits": list(reversed(slot_outputs)),
            "parent_logits": list(reversed(parent_outputs)),
            "separation_logits": list(reversed(separation_outputs)),
            "resolution_embedding": resolution_embedding,
        }
        if not self.decoder.deep_supervision:
            outputs = {
                key: value[0] if isinstance(value, list) else value
                for key, value in outputs.items()
            }
        if return_hierarchy:
            return outputs
        return outputs["semantic_logits"]

    def forward(
        self,
        x: torch.Tensor,
        return_hierarchy: bool = False,
        voxel_size: torch.Tensor | Sequence[float] | None = None,
    ):
        return self.decode_from_skips(
            self.encode(x), return_hierarchy=return_hierarchy, voxel_size=voxel_size
        )
