from __future__ import annotations

import os
from typing import List, Tuple, Union

import torch
from nnunet_mednext import create_mednext_v1

from .label_mapping_no_slot3 import NUM_DYNAMIC_GROUPS, NUM_OUTPUT_CHANNELS
from .network_structured_conditional_mednext import StructuredConditionalMedNeXt
from .trainer_structured_conditional_no_slot3 import nnUNetTrainerStructuredConditionalNoSlot3


class nnUNetTrainerStructuredConditionalNoSlot3MedNeXt(nnUNetTrainerStructuredConditionalNoSlot3):
    """
    No-slot3 structured-conditional trainer using MedNeXt as the backbone.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.mednext_model_id = str(os.environ.get("NNUNET_STRUCTCOND_MEDNEXT_MODEL_ID", "S")).strip().upper()
        self.mednext_kernel_size = int(os.environ.get("NNUNET_STRUCTCOND_MEDNEXT_KERNEL_SIZE", "3"))

        if self.mednext_model_id not in {"S", "B", "M", "L"}:
            raise ValueError("NNUNET_STRUCTCOND_MEDNEXT_MODEL_ID must be one of: S, B, M, L")
        if self.mednext_kernel_size not in {3, 5, 7}:
            raise ValueError("NNUNET_STRUCTCOND_MEDNEXT_KERNEL_SIZE must be one of: 3, 5, 7")

    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file(
            "[StructuredConditionalNoSlot3MedNeXt] "
            f"mednext_model_id={self.mednext_model_id}, "
            f"mednext_kernel_size={self.mednext_kernel_size}"
        )

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> torch.nn.Module:
        del architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import, num_output_channels
        mednext_model_id = str(os.environ.get("NNUNET_STRUCTCOND_MEDNEXT_MODEL_ID", "S")).strip().upper()
        mednext_kernel_size = int(os.environ.get("NNUNET_STRUCTCOND_MEDNEXT_KERNEL_SIZE", "3"))

        if mednext_model_id not in {"S", "B", "M", "L"}:
            raise ValueError("NNUNET_STRUCTCOND_MEDNEXT_MODEL_ID must be one of: S, B, M, L")
        if mednext_kernel_size not in {3, 5, 7}:
            raise ValueError("NNUNET_STRUCTCOND_MEDNEXT_KERNEL_SIZE must be one of: 3, 5, 7")

        backbone = create_mednext_v1(
            num_input_channels=int(num_input_channels),
            num_classes=int(NUM_OUTPUT_CHANNELS),
            model_id=mednext_model_id,
            kernel_size=int(mednext_kernel_size),
            # Keep DS heads instantiated for checkpoint compatibility during inference.
            deep_supervision=True,
        )
        # Matches usage in BANIS and helps reduce VRAM peaks for larger models.
        if hasattr(backbone, "outside_block_checkpointing"):
            backbone.outside_block_checkpointing = True
        return StructuredConditionalMedNeXt(
            backbone=backbone,
            num_groups=NUM_DYNAMIC_GROUPS,
            num_input_channels=int(num_input_channels),
            cond_dim=64,
        )
