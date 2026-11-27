"""AnchorSlot group-head recipe with the non-query foreground merged into background (reviewer ablation, 2026-10-27).

Identical to nnUNetTrainerStructuredConditionalERDynamicGroupHead except the structured target: voxels of the
dynamic labels of every non-queried group, which the paper model supervises as the explicit non-query channel
('other', channel 8), are supervised as background instead. Anchors, slots, validity masks, losses, sampling and the
group head (whose coarse target is query-independent) are unchanged. The 'other' channel stays in the network so the
architecture and checkpoint layout match; it never has a positive target and is driven towards zero.
"""
from __future__ import annotations

import torch

from .label_mapping_er_dynamic import BACKGROUND_CHANNEL, OTHER_CHANNEL
from .trainer_structured_conditional_er_dynamic_group_head import nnUNetTrainerStructuredConditionalERDynamicGroupHead


def merge_other_into_background(remapped: torch.Tensor) -> torch.Tensor:
    return remapped.masked_fill(remapped == OTHER_CHANNEL, BACKGROUND_CHANNEL)


class nnUNetTrainerStructuredConditionalERDynamicGroupHeadNQBG(nnUNetTrainerStructuredConditionalERDynamicGroupHead):
    """Group-head AnchorSlot with non-query foreground supervised as background."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        # explicit signature: nnUNetTrainer records init kwargs by inspecting it
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.print_to_log_file("[NQBG] non-query foreground (channel 'other') supervised as background")

    def _remap_target_for_group(self, target, group_ids: torch.Tensor):
        remapped, valid, active_slots = super()._remap_target_for_group(target, group_ids)
        if isinstance(remapped, list):
            return [merge_other_into_background(t) for t in remapped], valid, active_slots
        return merge_other_into_background(remapped), valid, active_slots
