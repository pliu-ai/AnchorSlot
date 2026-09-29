from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .label_mapping_er_dynamic import NUM_OUTPUT_CHANNELS
from .structured_loss_er_dynamic import StructuredConditionalLoss, StructuredLossConfig


class AnchorSlotDynamicDistributionLoss(StructuredConditionalLoss):
    """Original AnchorSlot loss with DDW applied only to masked CE.

    The supplied weights are per sample because conditional slot channels map
    to different original CellMap classes for different group IDs. Dice,
    conditional-focus, and suppression stay unchanged.
    """

    def __init__(self, config: Optional[StructuredLossConfig] = None) -> None:
        super().__init__(config)
        self._sample_class_weights: Optional[torch.Tensor] = None

    def set_sample_class_weights(self, weights: Optional[torch.Tensor]) -> None:
        if weights is not None:
            if weights.ndim != 2 or weights.shape[1] != NUM_OUTPUT_CHANNELS:
                raise ValueError(
                    "AnchorSlot DDW weights must have shape [B, NUM_OUTPUT_CHANNELS]"
                )
        self._sample_class_weights = weights

    def _masked_ce(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        active_conditional_slots: torch.Tensor,
    ) -> torch.Tensor:
        if self._sample_class_weights is None:
            return super()._masked_ce(logits, target, valid_mask, active_conditional_slots)

        logits_ce = self._mask_unused_cond3_channel_for_ce(
            logits, active_conditional_slots
        )
        target_labels = target[:, 0].long().clamp(min=0, max=NUM_OUTPUT_CHANNELS - 1)
        valid = valid_mask[:, 0].float()
        if not torch.any(valid > 0):
            return self._safe_zero(logits)

        sample_weights = self._sample_class_weights.to(
            device=logits.device, dtype=logits.dtype
        )
        if sample_weights.shape[0] != logits.shape[0]:
            raise ValueError(
                f"AnchorSlot DDW batch mismatch: {sample_weights.shape[0]} vs {logits.shape[0]}"
            )
        voxel_weights = torch.gather(sample_weights, 1, target_labels.flatten(1)).view_as(
            target_labels
        )
        weighted_valid = voxel_weights * valid
        ce_map = F.cross_entropy(logits_ce, target_labels, reduction="none")
        return (ce_map * weighted_valid).sum() / weighted_valid.sum().clamp_min(1.0)
