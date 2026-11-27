"""
Training-side ablations of the AnchorSlot ER-dynamic recipe. Each trainer changes exactly one thing
relative to nnUNetTrainerStructuredConditionalERDynamic (the E4-control recipe):

- QueryAwareFG:        force-foreground patches are centred on the queried group.
- DiceNonEmpty:        Dice entries with empty ground truth are dropped from the mean.
- RegionSuppress:      suppression averages background / anchor / non-query-foreground regions.
- FlatHeadSampled:     no conditioning; a 45-channel flat head, same one-query-per-sample training step.
- GroupedSoftmaxFlat:  the same flat head trained on all 13 group softmaxes every step (~4.7x step cost).
"""
from __future__ import annotations

from dataclasses import replace
import os
from typing import List, Tuple, Union

import numpy as np
import torch

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

from .grouped_softmax_flat_er_dynamic import (
    NUM_FLAT_OUTPUT_CHANNELS,
    GroupedSoftmaxFlatNet,
    grouped_softmax_loss,
    remap_target_fast,
)
from .trainer_structured_conditional_er_dynamic import nnUNetTrainerStructuredConditionalERDynamic

__all__ = [
    "nnUNetTrainerStructuredConditionalERDynamic",
    "nnUNetTrainerStructuredConditionalERDynamicQueryAwareFG",
    "nnUNetTrainerStructuredConditionalERDynamicDiceNonEmpty",
    "nnUNetTrainerStructuredConditionalERDynamicRegionSuppress",
    "nnUNetTrainerStructuredConditionalERDynamicFlatHeadSampled",
    "nnUNetTrainerStructuredConditionalERDynamicGroupedSoftmaxFlat",
]


class nnUNetTrainerStructuredConditionalERDynamicQueryAwareFG(nnUNetTrainerStructuredConditionalERDynamic):
    """A: centre force-foreground patches on a voxel of the queried group when it is present."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        # Explicit signature: nnUNetTrainer records its init kwargs from the signature for checkpoints.
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.query_aware_foreground = True


class nnUNetTrainerStructuredConditionalERDynamicDiceNonEmpty(nnUNetTrainerStructuredConditionalERDynamic):
    """B: exclude empty-ground-truth entries from the Dice and conditional-focus means."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        # Explicit signature: nnUNetTrainer records its init kwargs from the signature for checkpoints.
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.loss_cfg = replace(self.loss_cfg, dice_exclude_empty_gt=True)


class nnUNetTrainerStructuredConditionalERDynamicRegionSuppress(nnUNetTrainerStructuredConditionalERDynamic):
    """C: region-balanced selected-slot suppression."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        # Explicit signature: nnUNetTrainer records its init kwargs from the signature for checkpoints.
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.loss_cfg = replace(self.loss_cfg, suppression_mode="region_balanced")


class nnUNetTrainerStructuredConditionalERDynamicFlatHeadSampled(nnUNetTrainerStructuredConditionalERDynamic):
    """
    D1: the same target code, losses, one-query-per-sample sampling, data pipeline and 9-channel
    inference interface, but without FiLM conditioning: each group's slot and `other` logits are
    group-specific channels of one shared, unconditioned 45-channel head. Isolates conditioning.
    """

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> torch.nn.Module:
        del num_output_channels
        backbone = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            NUM_FLAT_OUTPUT_CHANNELS,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        print(
            f"[GroupedSoftmaxFlat] unconditioned backbone with {NUM_FLAT_OUTPUT_CHANNELS} flat output channels",
            flush=True,
        )
        return GroupedSoftmaxFlatNet(backbone)


class nnUNetTrainerStructuredConditionalERDynamicGroupedSoftmaxFlat(nnUNetTrainerStructuredConditionalERDynamicFlatHeadSampled):
    """D13: the D1 network, but every step supervises all 13 group softmaxes (mean of 13 structured losses)."""

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
        # Recompute each group's loss in backward (keeps activation memory near one group's); 0 disables.
        self.grouped_loss_checkpoint = str(
            os.environ.get("NNUNET_STRUCTCOND_GROUPED_LOSS_CHECKPOINT", "1")
        ).lower() in {"1", "true", "yes", "y"}

    def _remap_target_fast(self, target, group_ids: torch.Tensor):
        return remap_target_fast(target, group_ids, ignore_label=self.label_manager.ignore_label)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._to_device_target(batch["target"], self.device)
        batch_size = int(data.shape[0])
        self._group_sample_counter_epoch += batch_size  # every group is supervised for every sample

        self.network.zero_grad(set_to_none=True)
        with self._autocast_context():
            output = self.network(data, None, all_groups=True)
        self._assert_all_finite(output, "network output")

        loss_output = self._output_to_float(output) if self.loss_in_fp32 else output
        loss = grouped_softmax_loss(
            loss_output,
            target,
            self._compute_structured_loss,
            self._remap_target_fast,
            num_groups=self.num_dynamic_groups,
            use_checkpoint=bool(self.grouped_loss_checkpoint),
        )
        self._assert_all_finite(loss, "training loss")

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self._get_optimizer_parameters(), self.grad_clip_norm)
            if not self._all_finite(grad_norm):
                return self._skip_nonfinite_gradient_update(loss, grad_norm)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self._get_optimizer_parameters(), self.grad_clip_norm)
            if not self._all_finite(grad_norm):
                return self._skip_nonfinite_gradient_update(loss, grad_norm)
            self.optimizer.step()

        loss_value = loss.detach().cpu().numpy()
        # Same key set as the base train_step so on_train_epoch_end can collate outputs.
        return {
            "loss": loss_value,
            "structured_loss": loss_value,
            "text_contrast_loss": np.zeros_like(loss_value),
            "text_contrast_num_samples": 0,
            "text_contrast_num_labels": 0,
            "text_contrast_sampled_label_ids": [],
            "text_contrast_missing_label_ids": [],
            "text_generated_slot_active_slots": 0,
            "text_generated_slot_inactive_slots": 0,
            "text_generated_slot_logit_mean": 0.0,
            "text_generated_slot_logit_std": 0.0,
            "text_generated_slot_logit_min": 0.0,
            "text_generated_slot_logit_max": 0.0,
        }
