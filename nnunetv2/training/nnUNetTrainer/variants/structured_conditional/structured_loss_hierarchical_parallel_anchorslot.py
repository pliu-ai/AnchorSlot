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
    def _masked_ce(logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        target_labels = target[:, 0].long()
        valid = valid_mask[:, 0]
        if not torch.any(valid):
            return logits.sum() * 0.0
        ce = F.cross_entropy(logits, target_labels, reduction="none")
        return ce[valid].mean()

    def _semantic_dice(
        self,
        semantic_logits: torch.Tensor,
        semantic_target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
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
    ):
        if semantic_logits.shape[1] != NUM_ORIGINAL_CLASSES:
            raise ValueError("semantic logits have an invalid channel count")
        if coarse_logits.shape[1] != NUM_COARSE_CHANNELS:
            raise ValueError("coarse logits have an invalid channel count")

        semantic, coarse, group, slot, valid = build_hierarchical_targets(
            segmentation,
            ignore_label=ignore_label,
        )
        semantic_ce = self._masked_ce(semantic_logits, semantic, valid)
        semantic_dice = self._semantic_dice(semantic_logits, semantic, valid)
        coarse_ce = self._masked_ce(coarse_logits, coarse, valid)
        slot_ce = self._slot_ce(slot_logits, group, slot, valid)

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
