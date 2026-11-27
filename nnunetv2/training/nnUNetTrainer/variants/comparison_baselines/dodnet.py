"""DoDNet's image/task-conditioned dynamic head on a shared nnU-Net backbone.

Independent implementation of the mechanism in Zhang et al., CVPR 2021,
https://arxiv.org/abs/2011.10217 (sections 3.3--3.6). Architecture checked against
https://github.com/jianpengz/DoDNet/blob/main/a_DynConv/unet3D_DynConv882.py.

The 8->8->8->K head, one-hot task encoding, single-layer controller, and binary
outputs follow DoDNet. PlainConvUNet, per-scale deep supervision, K=3 and the
17 CellMap tasks are explicit benchmark adaptations, not the original model.
There is no conditioning of encoder/decoder features and no anchor branch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from .taxonomy import DYNAMIC_GROUP_LABELS, DYNAMIC_GROUP_NAMES, LEAF_NAMES, SINGLETON_LABELS


# Fixed benchmark contract, deliberately independent of AnchorSlot environment
# variables and mutable grouping files. The last four tasks are ordinary tasks.
DEFAULT_DODNET_TASK_LABELS: Tuple[Tuple[int, ...], ...] = DYNAMIC_GROUP_LABELS + tuple((i,) for i in SINGLETON_LABELS)
DEFAULT_DODNET_TASK_NAMES = DYNAMIC_GROUP_NAMES + tuple(LEAF_NAMES[i] for i in SINGLETON_LABELS)


def _group_count(channels: int, requested: int) -> int:
    # Keep at least two channels per group: B=1 and a 1^3 bottleneck must work.
    return next(g for g in range(min(requested, max(1, channels // 2)), 0, -1)
                if channels % g == 0)


def _task_ids(task_ids: torch.Tensor, batch: int, num_tasks: int,
              device: torch.device) -> torch.Tensor:
    ids = torch.as_tensor(task_ids, device=device)
    if ids.ndim != 1 or ids.shape[0] != batch:
        raise ValueError(f"task_ids must have shape [{batch}], got {tuple(ids.shape)}")
    if ids.dtype == torch.bool or ids.is_floating_point() or ids.is_complex():
        raise ValueError("task_ids must be integers")
    ids = ids.to(dtype=torch.long)
    if bool(((ids < 0) | (ids >= num_tasks)).any()):
        raise ValueError(f"task_ids must be in [0, {num_tasks})")
    return ids


def _validate_task_labels(task_labels: Sequence[Sequence[int]], out_channels: int,
                          num_labels: int) -> Tuple[Tuple[int, ...], ...]:
    labels = tuple(tuple(int(label) for label in task) for task in task_labels)
    if not labels or any(not task or len(task) > out_channels for task in labels):
        raise ValueError("each task must contain 1..out_channels foreground labels")
    flat = [label for task in labels for label in task]
    if sorted(flat) != list(range(1, num_labels)):
        raise ValueError("task labels must cover every foreground ID exactly once")
    return labels


def merge_task_logits_to_full_scores(
    task_logits: Sequence[torch.Tensor],
    task_labels: Sequence[Sequence[int]] = DEFAULT_DODNET_TASK_LABELS,
    num_labels: int = 32,
) -> torch.Tensor:
    """Map independent binary logits to original label order, with zero bg score.

    Argmax selects the largest foreground sigmoid if it exceeds 0.5; otherwise
    background wins (including exact ties at zero). These scores are NOT logits
    of a trained categorical distribution. A subsequent softmax preserves argmax
    but does not yield calibrated foreground probabilities. Unused slots never
    participate, regardless of their value.
    """
    if len(task_logits) != len(task_labels) or not task_logits:
        raise ValueError("one logit tensor is required for every task")
    first = task_logits[0]
    if first.ndim != 5 or not first.is_floating_point():
        raise ValueError("task logits must be floating [B, K, D, H, W] tensors")
    labels = _validate_task_labels(task_labels, first.shape[1], num_labels)
    foreground = {}
    for logits, local_labels in zip(task_logits, labels):
        if logits.shape != first.shape or logits.device != first.device or logits.dtype != first.dtype:
            raise ValueError("all task logit tensors must have identical shape, dtype and device")
        for slot, label in enumerate(local_labels):
            foreground[label] = logits[:, slot:slot + 1]
    return torch.cat([torch.zeros_like(first[:, :1])] +
                     [foreground[label] for label in range(1, num_labels)], dim=1)


class DoDNetDynamicHead(nn.Module):
    """Three per-sample 1x1x1 convolutions, generated from image and task.

    For hidden_channels=8 and out_channels=2 the controller generates exactly
    162 scalars. Kernels remain tensors in the autograd graph, never Parameters
    created during forward. Grouped convolution isolates each batch example.
    """

    def __init__(self, in_channels: int, context_channels: int, num_tasks: int,
                 out_channels: int, hidden_channels: int = 8,
                 norm_groups: int = 16) -> None:
        super().__init__()
        if min(in_channels, context_channels, num_tasks, out_channels,
               hidden_channels, norm_groups) <= 0:
            raise ValueError("channel counts, task count and norm_groups must be positive")
        self.in_channels = int(in_channels)
        self.context_channels = int(context_channels)
        self.num_tasks = int(num_tasks)
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        h = self.hidden_channels
        self.weight_nums = (h * h, h * h, h * self.out_channels)
        self.bias_nums = (h, h, self.out_channels)
        self.num_dynamic_params = sum(self.weight_nums) + sum(self.bias_nums)
        self.precls = nn.Sequential(
            nn.GroupNorm(_group_count(self.in_channels, norm_groups), self.in_channels),
            nn.ReLU(inplace=False), nn.Conv3d(self.in_channels, h, 1),
        )
        self.context_pool = nn.Sequential(
            nn.GroupNorm(_group_count(self.context_channels, norm_groups), self.context_channels),
            nn.ReLU(inplace=False), nn.AdaptiveAvgPool3d(1),
        )
        self.controller = nn.Conv3d(self.context_channels + self.num_tasks,
                                    self.num_dynamic_params, 1)

    def prepare_features(self, features: torch.Tensor, context: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 5 or features.shape[1] != self.in_channels:
            raise ValueError("features must be [B, in_channels, D, H, W]")
        if context.ndim != 5 or context.shape[1] != self.context_channels:
            raise ValueError("context must be [B, context_channels, D, H, W]")
        if features.shape[0] != context.shape[0] or features.device != context.device:
            raise ValueError("features and context must share batch size and device")
        return self.precls(features), self.context_pool(context)

    def generate_parameters(self, image_context: torch.Tensor,
                            task_ids: torch.Tensor) -> torch.Tensor:
        if image_context.ndim != 5 or image_context.shape[1:] != (self.context_channels, 1, 1, 1):
            raise ValueError("image_context must be pooled [B, context_channels, 1, 1, 1]")
        ids = _task_ids(task_ids, image_context.shape[0], self.num_tasks, image_context.device)
        one_hot = F.one_hot(ids, self.num_tasks).to(dtype=image_context.dtype)
        conditioned = torch.cat((image_context, one_hot[:, :, None, None, None]), dim=1)
        return self.controller(conditioned).flatten(1)

    def apply_dynamic_parameters(self, features: torch.Tensor,
                                 parameters: torch.Tensor) -> torch.Tensor:
        if features.ndim != 5 or features.shape[1] != self.hidden_channels or features.shape[0] == 0:
            raise ValueError("projected features must be nonempty [B, hidden_channels, D, H, W]")
        batch, _, depth, height, width = features.shape
        if parameters.shape != (batch, self.num_dynamic_params):
            raise ValueError("dynamic parameter shape does not match batch and head")
        if parameters.device != features.device:
            raise ValueError("dynamic parameters and features must share a device")
        pieces = parameters.split(self.weight_nums + self.bias_nums, dim=1)
        value = features.reshape(1, batch * self.hidden_channels, depth, height, width)
        for layer, output_channels in enumerate((self.hidden_channels, self.hidden_channels,
                                                self.out_channels)):
            weight = pieces[layer].reshape(batch * output_channels, self.hidden_channels, 1, 1, 1)
            bias = pieces[layer + 3].reshape(batch * output_channels)
            value = F.conv3d(value, weight, bias, groups=batch)
            if layer < 2:
                value = F.relu(value)
        return value.reshape(batch, self.out_channels, depth, height, width)

    def forward(self, features: torch.Tensor, context: torch.Tensor,
                task_ids: torch.Tensor) -> torch.Tensor:
        projected, pooled = self.prepare_features(features, context)
        return self.apply_dynamic_parameters(projected, self.generate_parameters(pooled, task_ids))


@dataclass(frozen=True)
class DoDNetFeatures:
    """Call-local cache: projected shared features and pooled image descriptors."""

    projected: Tuple[torch.Tensor, ...]
    contexts: Tuple[torch.Tensor, ...]
    stage_indices: Tuple[int, ...]
    deep_supervision: bool


class DoDNetUNet(nn.Module):
    """Reuse a PlainConvUNet encoder/decoder, replacing all static output heads.

    Explicit task IDs return K independent binary logits. With task_ids=None,
    evaluate every task from one shared feature extraction and return full-label
    scores. Both modes follow decoder.deep_supervision (highest resolution first).
    Deep supervision adds one dynamic head/controller per decoder output scale.
    """

    def __init__(self, backbone: nn.Module, num_tasks: int = 17, out_channels: int = 3,
                 *, task_labels: Sequence[Sequence[int]] = DEFAULT_DODNET_TASK_LABELS,
                 hidden_channels: int = 8, norm_groups: int = 16,
                 deep_supervision: Optional[bool] = None) -> None:
        super().__init__()
        self.num_tasks = int(num_tasks)
        self.out_channels = int(out_channels)
        self.num_output_channels = 1 + sum(len(task) for task in task_labels)
        self.task_labels = _validate_task_labels(task_labels, out_channels, self.num_output_channels)
        if len(self.task_labels) != self.num_tasks:
            raise ValueError("num_tasks must equal the number of task label groups")
        decoder = backbone.decoder
        channels = [layer.in_channels for layer in decoder.seg_layers]
        if not channels or len(channels) != len(decoder.stages):
            raise ValueError("backbone must expose one segmentation head per decoder stage")
        if any(not isinstance(layer, nn.Conv3d) for layer in decoder.seg_layers):
            raise ValueError("DoDNetUNet requires a 3D convolutional backbone")
        context_channels = int(backbone.encoder.output_channels[-1])
        self.backbone = backbone
        # Replace instead of retaining unused trainable static segmentation layers.
        decoder.seg_layers = nn.ModuleList([
            DoDNetDynamicHead(c, context_channels, num_tasks, out_channels,
                              hidden_channels, norm_groups) for c in channels
        ])
        if deep_supervision is not None:
            decoder.deep_supervision = bool(deep_supervision)

    @property
    def encoder(self) -> nn.Module:
        return self.backbone.encoder

    @property
    def decoder(self) -> nn.Module:
        return self.backbone.decoder

    @property
    def heads(self) -> nn.ModuleList:
        return self.decoder.seg_layers

    def extract_features(self, x: torch.Tensor) -> DoDNetFeatures:
        skips = self.encoder(x)
        if not isinstance(skips, (tuple, list)):
            raise ValueError("backbone encoder must return all skip features")
        context = skips[-1]
        value = context
        projected, contexts, stages = [], [], []
        deep_supervision = bool(self.decoder.deep_supervision)
        for stage, block in enumerate(self.decoder.stages):
            value = self.decoder.transpconvs[stage](value)
            value = block(torch.cat((value, skips[-stage - 2]), dim=1))
            if deep_supervision or stage == len(self.decoder.stages) - 1:
                precls, pooled = self.heads[stage].prepare_features(value, context)
                projected.append(precls)
                contexts.append(pooled)
                stages.append(stage)
        return DoDNetFeatures(tuple(reversed(projected)), tuple(reversed(contexts)),
                              tuple(reversed(stages)), deep_supervision)

    def predict_from_features(self, features: DoDNetFeatures,
                              task_ids: Optional[torch.Tensor] = None
                              ) -> Union[torch.Tensor, list]:
        outputs = []
        for projected, context, stage in zip(features.projected, features.contexts,
                                              features.stage_indices):
            head = self.heads[stage]
            if task_ids is not None:
                logits = head.apply_dynamic_parameters(projected, head.generate_parameters(context, task_ids))
            else:
                task_logits = []
                for task in range(self.num_tasks):
                    ids = torch.full((projected.shape[0],), task, dtype=torch.long, device=projected.device)
                    task_logits.append(head.apply_dynamic_parameters(projected, head.generate_parameters(context, ids)))
                logits = merge_task_logits_to_full_scores(task_logits, self.task_labels, self.num_output_channels)
            outputs.append(logits)
        return outputs if features.deep_supervision else outputs[0]

    def forward(self, x: torch.Tensor, task_ids: Optional[torch.Tensor] = None
                ) -> Union[torch.Tensor, list]:
        return self.predict_from_features(self.extract_features(x), task_ids)
