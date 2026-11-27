"""Sampling-only variant of the group-head AnchorSlot trainer (sampling_f0_20261026).

The paper recipe samples the query group per crop, but centres a foreground-forced patch on a random class present in
the crop, so a queried group is often missing from the final patch (exposure probe 20261026: 39-58% kept for small
classes). Here the forced centre is a voxel of the queried group when the crop has it (the existing
StructuredConditionalDataLoader3D option query_aware_foreground). Everything else is the parent trainer; the group
sampling mode (random / balanced) stays an env choice (NNUNET_STRUCTCOND_GROUP_SAMPLING_MODE).
"""
from __future__ import annotations

import torch

from .trainer_structured_conditional_er_dynamic_group_head import nnUNetTrainerStructuredConditionalERDynamicGroupHead


class nnUNetTrainerStructuredConditionalERDynamicGroupHeadQAF(nnUNetTrainerStructuredConditionalERDynamicGroupHead):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        # explicit signature: nnUNetTrainer records init kwargs by inspecting it
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.query_aware_foreground = True
        self.print_to_log_file(f"[QAF] query-aware foreground on; group sampling mode {self.group_sampling_mode}, "
                               f"balance power {self.group_balance_power}")
