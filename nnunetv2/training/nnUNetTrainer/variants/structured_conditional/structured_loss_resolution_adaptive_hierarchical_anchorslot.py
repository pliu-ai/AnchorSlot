from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .hierarchical_parallel_mapping import NUM_ORIGINAL_CLASSES, build_hierarchical_targets
from .resolution_adaptive_mapping import (
    NUM_PARENT_CLASSES,
    active_parent_mask,
    build_instance_separation_targets,
    build_parent_targets,
    derived_parent_probabilities,
)
from .structured_loss_hierarchical_parallel_anchorslot import (
    HierarchicalParallelAnchorSlotLoss,
    HierarchicalParallelLossConfig,
)


@dataclass
class ResolutionAdaptiveLossConfig(HierarchicalParallelLossConfig):
    lambda_parent_bce: float = 0.5
    lambda_hierarchy_consistency: float = 0.25
    lambda_boundary: float = 0.1
    lambda_affinity: float = 0.1
    ignore_unannotated_background: bool = False


class ResolutionAdaptiveHierarchicalAnchorSlotLoss(HierarchicalParallelAnchorSlotLoss):
    """Leaf/slot loss plus official DAG, hierarchy, and separation supervision."""

    def __init__(self, config: Optional[ResolutionAdaptiveLossConfig] = None) -> None:
        self.ra_config = config if config is not None else ResolutionAdaptiveLossConfig()
        super().__init__(self.ra_config)

    @staticmethod
    def _masked_binary_loss(
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.any(mask):
            return logits.sum() * 0.0
        values = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return values[mask].mean()

    def forward(
        self,
        semantic_logits: torch.Tensor,
        coarse_logits: torch.Tensor,
        slot_logits: torch.Tensor,
        parent_logits: torch.Tensor,
        separation_logits: torch.Tensor,
        segmentation: torch.Tensor,
        ignore_label: Optional[int] = None,
        active_semantic_mask: Optional[torch.Tensor] = None,
        active_parent_annotation_mask: Optional[torch.Tensor] = None,
        instance_ids: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ):
        if parent_logits.shape[1] != NUM_PARENT_CLASSES:
            raise ValueError("parent logits have an invalid channel count")
        base_total, base_components = super().forward(
            semantic_logits,
            coarse_logits,
            slot_logits,
            segmentation,
            ignore_label=ignore_label,
            return_components=True,
            active_semantic_mask=active_semantic_mask,
            ignore_unannotated_background=self.ra_config.ignore_unannotated_background,
        )

        semantic, _, _, _, valid = build_hierarchical_targets(
            segmentation, ignore_label=ignore_label
        )
        parent_valid = valid.clone()
        parent_active = None
        if active_semantic_mask is not None:
            active_semantic_mask = active_semantic_mask.to(semantic_logits.device).bool()
            active_semantic_mask = active_semantic_mask.clone()
            active_semantic_mask[:, 0] = True
            target_active = active_semantic_mask.gather(
                1, semantic[:, 0].flatten(1)
            ).view_as(semantic[:, 0])
            valid = valid & target_active[:, None]
            parent_active = active_parent_mask(active_semantic_mask)
            if self.ra_config.ignore_unannotated_background:
                partial = active_semantic_mask[:, 1:].sum(dim=1) < (NUM_ORIGINAL_CLASSES - 1)
                background = semantic[:, 0] == 0
                partial_shape = (-1, *([1] * (background.ndim - 1)))
                valid = valid & ~(
                    partial.view(partial_shape) & background
                )[:, None]
        if active_parent_annotation_mask is not None:
            explicit_parent = active_parent_annotation_mask.to(parent_logits.device).bool()
            if explicit_parent.shape != (parent_logits.shape[0], NUM_PARENT_CLASSES):
                raise ValueError(
                    f"active_parent_annotation_mask must have shape "
                    f"[{parent_logits.shape[0]}, {NUM_PARENT_CLASSES}]"
                )
            parent_active = explicit_parent

        # Parent-only low-resolution annotations are often encoded by a
        # representative descendant ID in the merged nnU-Net segmentation. They
        # must supervise the parent head even when that atomic descendant is not
        # itself annotated and is therefore masked from the leaf loss.
        parent_target = build_parent_targets(semantic, parent_valid)
        parent_mask = parent_valid.expand(
            -1, NUM_PARENT_CLASSES, *([-1] * (parent_valid.ndim - 2))
        )
        if parent_active is not None:
            parent_shape = (
                parent_active.shape[0], parent_active.shape[1],
                *([1] * (parent_valid.ndim - 2)),
            )
            parent_mask = parent_mask & parent_active.view(parent_shape)
        parent_bce = self._masked_binary_loss(parent_logits, parent_target, parent_mask)

        derived_parent = derived_parent_probabilities(semantic_logits)
        direct_parent = torch.sigmoid(parent_logits)
        if torch.any(parent_mask):
            hierarchy_consistency = F.smooth_l1_loss(
                direct_parent[parent_mask], derived_parent[parent_mask]
            )
        else:
            hierarchy_consistency = parent_logits.sum() * 0.0

        separation_target, separation_mask = build_instance_separation_targets(
            semantic, valid, instance_ids=instance_ids
        )
        if separation_target.shape != separation_logits.shape:
            raise ValueError(
                f"separation output/target mismatch: {separation_logits.shape} vs {separation_target.shape}"
            )
        separation_values = F.binary_cross_entropy_with_logits(
            separation_logits, separation_target, reduction="none"
        )
        boundary_mask = separation_mask[:, :1]
        affinity_mask = separation_mask[:, 1:]
        boundary = (
            separation_values[:, :1][boundary_mask].mean()
            if torch.any(boundary_mask)
            else separation_logits.sum() * 0.0
        )
        affinity = (
            separation_values[:, 1:][affinity_mask].mean()
            if torch.any(affinity_mask)
            else separation_logits.sum() * 0.0
        )

        total = (
            base_total
            + self.ra_config.lambda_parent_bce * parent_bce
            + self.ra_config.lambda_hierarchy_consistency * hierarchy_consistency
            + self.ra_config.lambda_boundary * boundary
            + self.ra_config.lambda_affinity * affinity
        )
        if not return_components:
            return total
        components: Dict[str, torch.Tensor] = dict(base_components)
        components.update(
            {
                "parent_bce": parent_bce.detach(),
                "hierarchy_consistency": hierarchy_consistency.detach(),
                "boundary": boundary.detach(),
                "affinity": affinity.detach(),
                "total": total.detach(),
            }
        )
        return total, components
