from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .hierarchical_parallel_mapping import (
    NUM_COARSE_CHANNELS,
    NUM_ORIGINAL_CLASSES,
    build_hierarchical_targets,
    hierarchy_active_masks,
)


@dataclass
class HierarchicalParallelLossConfig:
    lambda_semantic_ce: float = 1.0
    lambda_semantic_dice: float = 1.0
    lambda_coarse_ce: float = 0.5
    lambda_slot_ce: float = 0.5
    smooth: float = 1e-5


class HierarchicalParallelAnchorSlotLoss(nn.Module):
    """Joint leaf, coarse-group, and within-group anchor-slot supervision."""

    def __init__(self, config: Optional[HierarchicalParallelLossConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else HierarchicalParallelLossConfig()

    @staticmethod
    def _masked_ce(
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        active_channels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target_labels = target[:, 0].long()
        valid = valid_mask[:, 0]
        if active_channels is not None:
            active_channels = active_channels.bool()
            active_for_target = active_channels.gather(1, target_labels.flatten(1)).view_as(target_labels)
            valid = valid & active_for_target
            channel_shape = (active_channels.shape[0], active_channels.shape[1], *([1] * (logits.ndim - 2)))
            logits = logits.masked_fill(~active_channels.view(channel_shape), -1e4)
        if not torch.any(valid):
            return logits.sum() * 0.0
        ce = F.cross_entropy(logits, target_labels, reduction="none")
        return ce[valid].mean()

    def _semantic_dice(
        self,
        semantic_logits: torch.Tensor,
        semantic_target: torch.Tensor,
        valid_mask: torch.Tensor,
        active_semantic: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if active_semantic is not None:
            channel_shape = (
                active_semantic.shape[0], active_semantic.shape[1],
                *([1] * (semantic_logits.ndim - 2)),
            )
            semantic_logits = semantic_logits.masked_fill(
                ~active_semantic.bool().view(channel_shape), -1e4
            )
        probs = torch.softmax(semantic_logits, dim=1)
        labels = semantic_target[:, 0].long()
        onehot = F.one_hot(labels, num_classes=NUM_ORIGINAL_CLASSES)
        onehot = onehot.permute(0, -1, *range(1, onehot.ndim - 1)).float()
        valid = valid_mask.float()
        axes = (0, *range(2, probs.ndim))
        intersection = (probs * onehot * valid).sum(dim=axes)
        denominator = ((probs + onehot) * valid).sum(dim=axes)
        target_mass = (onehot * valid).sum(dim=axes)
        dice = (2.0 * intersection + self.config.smooth) / (
            denominator + self.config.smooth
        ).clamp_min(1e-8)

        # Background is excluded; classes absent in the batch do not dilute the
        # fine loss, while false-positive probability still activates a class.
        active = target_mass[1:] > 0
        if not torch.any(active):
            return semantic_logits.sum() * 0.0
        return 1.0 - dice[1:][active].mean()

    @staticmethod
    def _slot_ce(
        slot_logits: torch.Tensor,
        group_target: torch.Tensor,
        slot_target: torch.Tensor,
        valid_mask: torch.Tensor,
        active_slots: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dynamic = valid_mask[:, 0] & (group_target[:, 0] >= 0)
        if not torch.any(dynamic):
            return slot_logits.sum() * 0.0

        # [B, G, S, ...] -> [B, ..., G, S], then select each voxel's group.
        spatial_dims = list(range(3, slot_logits.ndim))
        permuted = slot_logits.permute(0, *spatial_dims, 1, 2)
        selected_logits = permuted[dynamic]
        selected_groups = group_target[:, 0][dynamic].long()
        row_ids = torch.arange(selected_groups.numel(), device=slot_logits.device)
        selected_logits = selected_logits[row_ids, selected_groups]
        if active_slots is not None:
            spatial_shape = group_target.shape[2:]
            expanded = active_slots.view(
                active_slots.shape[0], *([1] * len(spatial_shape)),
                active_slots.shape[1], active_slots.shape[2],
            ).expand(active_slots.shape[0], *spatial_shape, *active_slots.shape[1:])
            selected_active = expanded[dynamic][row_ids, selected_groups]
            selected_logits = selected_logits.masked_fill(~selected_active, -1e4)
        selected_slots = slot_target[:, 0][dynamic].long()
        return F.cross_entropy(selected_logits, selected_slots)

    def forward(
        self,
        semantic_logits: torch.Tensor,
        coarse_logits: torch.Tensor,
        slot_logits: torch.Tensor,
        segmentation: torch.Tensor,
        ignore_label: Optional[int] = None,
        return_components: bool = False,
        active_semantic_mask: Optional[torch.Tensor] = None,
        ignore_unannotated_background: bool = False,
    ):
        if semantic_logits.shape[1] != NUM_ORIGINAL_CLASSES:
            raise ValueError("semantic logits have an invalid channel count")
        if coarse_logits.shape[1] != NUM_COARSE_CHANNELS:
            raise ValueError("coarse logits have an invalid channel count")

        semantic, coarse, group, slot, valid = build_hierarchical_targets(
            segmentation,
            ignore_label=ignore_label,
        )
        active_coarse = active_slots = None
        if active_semantic_mask is not None:
            active_semantic_mask = active_semantic_mask.to(semantic_logits.device).bool().clone()
            active_semantic_mask[:, 0] = True
            active_coarse, active_slots = hierarchy_active_masks(active_semantic_mask)
            target_active = active_semantic_mask.gather(
                1, semantic[:, 0].flatten(1)
            ).view_as(semantic[:, 0])
            valid = valid & target_active[:, None]
            if ignore_unannotated_background:
                partial = active_semantic_mask[:, 1:].sum(dim=1) < (NUM_ORIGINAL_CLASSES - 1)
                background = semantic[:, 0] == 0
                valid = valid & ~(partial.view(-1, *([1] * (background.ndim - 1))) & background)[:, None]
        semantic_ce = self._masked_ce(
            semantic_logits, semantic, valid, active_semantic_mask
        )
        semantic_dice = self._semantic_dice(
            semantic_logits, semantic, valid, active_semantic_mask
        )
        coarse_ce = self._masked_ce(coarse_logits, coarse, valid, active_coarse)
        slot_ce = self._slot_ce(slot_logits, group, slot, valid, active_slots)

        total = (
            self.config.lambda_semantic_ce * semantic_ce
            + self.config.lambda_semantic_dice * semantic_dice
            + self.config.lambda_coarse_ce * coarse_ce
            + self.config.lambda_slot_ce * slot_ce
        )
        if not return_components:
            return total
        components: Dict[str, torch.Tensor] = {
            "semantic_ce": semantic_ce.detach(),
            "semantic_dice": semantic_dice.detach(),
            "coarse_ce": coarse_ce.detach(),
            "slot_ce": slot_ce.detach(),
            "total": total.detach(),
        }
        return total, components
