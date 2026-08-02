from __future__ import annotations

import os
from typing import List, Tuple, Union

import torch
from nnunet_mednext import create_mednext_v1

from .label_mapping_er_dynamic import NUM_DYNAMIC_GROUPS, NUM_OUTPUT_CHANNELS
from .network_structured_conditional_mednext import StructuredConditionalMedNeXt
from .trainer_structured_conditional_er_dynamic import nnUNetTrainerStructuredConditionalERDynamic


class nnUNetTrainerStructuredConditionalERDynamicMedNeXt(nnUNetTrainerStructuredConditionalERDynamic):
    """
    ER-dynamic structured-conditional trainer using MedNeXt as the backbone.

    Identical training logic to nnUNetTrainerStructuredConditionalERDynamic but
    replaces the nnU-Net plain-conv backbone with MedNeXt (create_mednext_v1).
    Conditioning is applied at input feature level via FiLM-style modulation
    (StructuredConditionalMedNeXt) since MedNeXt does not expose the same
    encoder/decoder skip internals as the default nnU-Net backbones.

    Environment variables (in addition to those inherited from the base trainer):
        NNUNET_STRUCTCOND_MEDNEXT_MODEL_ID   : S | B | M | L  (default: S)
        NNUNET_STRUCTCOND_MEDNEXT_KERNEL_SIZE: 3 | 5 | 7      (default: 3)
    """

    default_amp_dtype = "bfloat16"

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
            "[StructuredConditionalERDynamicMedNeXt] "
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
        # Plans-level arch kwargs are not used: MedNeXt is configured via env vars.
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
            # Always instantiate DS heads for checkpoint compatibility at inference.
            deep_supervision=True,
        )
        if hasattr(backbone, "do_ds"):
            backbone.do_ds = bool(enable_deep_supervision)
        # Gradient checkpointing reduces VRAM peaks for larger MedNeXt variants.
        if hasattr(backbone, "outside_block_checkpointing"):
            backbone.outside_block_checkpointing = True

        return StructuredConditionalMedNeXt(
            backbone=backbone,
            num_groups=NUM_DYNAMIC_GROUPS,
            num_input_channels=int(num_input_channels),
            cond_dim=64,
        )
