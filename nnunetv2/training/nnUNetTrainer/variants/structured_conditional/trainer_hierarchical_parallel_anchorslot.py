from __future__ import annotations

import json
import os
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context

from .hierarchical_parallel_mapping import (
    NUM_COARSE_CHANNELS,
    NUM_ORIGINAL_CLASSES,
)
from .inference_hierarchical_parallel_anchorslot import (
    predict_hierarchical_parallel,
    predict_original_labels_hierarchical_parallel,
)
from .network_hierarchical_parallel_anchorslot import (
    HierarchicalParallelAnchorSlotUNet,
    get_main_semantic_output,
)
from .structured_loss_hierarchical_parallel_anchorslot import (
    HierarchicalParallelAnchorSlotLoss,
    HierarchicalParallelLossConfig,
)
from .trainer_structured_conditional_no_slot3 import nnUNetTrainerStructuredConditionalNoSlot3


class nnUNetTrainerHierarchicalParallelAnchorSlot(nnUNetTrainerStructuredConditionalNoSlot3):
    """Train all 12 dynamic groups and both anchor slots in one forward pass."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.fixed_num_output_channels = NUM_ORIGINAL_CLASSES
        self.hpa_code_dim = int(os.environ.get("NNUNET_HPA_CODE_DIM", "64"))
        if self.hpa_code_dim < 1:
            raise ValueError("NNUNET_HPA_CODE_DIM must be >= 1")
        self.hpa_loss_cfg = HierarchicalParallelLossConfig(
            lambda_semantic_ce=float(os.environ.get("NNUNET_HPA_LAMBDA_SEMANTIC_CE", "1.0")),
            lambda_semantic_dice=float(os.environ.get("NNUNET_HPA_LAMBDA_SEMANTIC_DICE", "1.0")),
            lambda_coarse_ce=float(os.environ.get("NNUNET_HPA_LAMBDA_COARSE_CE", "0.5")),
            lambda_slot_ce=float(os.environ.get("NNUNET_HPA_LAMBDA_SLOT_CE", "0.5")),
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
        del num_output_channels
        backbone = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            NUM_COARSE_CHANNELS,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        code_dim = int(os.environ.get("NNUNET_HPA_CODE_DIM", "64"))
        return HierarchicalParallelAnchorSlotUNet(backbone=backbone, code_dim=code_dim)

    def _build_loss(self):
        return HierarchicalParallelAnchorSlotLoss(self.hpa_loss_cfg)

    def get_dataloaders(self):
        # Conditions are no longer sampled: every group is supervised in parallel.
        return nnUNetTrainer.get_dataloaders(self)

    @staticmethod
    def _sanitize_hierarchy(output):
        sanitized = {}
        for key, value in output.items():
            if isinstance(value, (tuple, list)):
                sanitized[key] = [torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4) for x in value]
            else:
                sanitized[key] = torch.nan_to_num(value, nan=0.0, posinf=1e4, neginf=-1e4)
        return sanitized

    def _compute_hierarchical_loss(self, output: dict, target, main_only: bool = False) -> torch.Tensor:
        semantic = output["semantic_logits"]
        coarse = output["coarse_logits"]
        slots = output["slot_logits"]

        if isinstance(semantic, (tuple, list)):
            if not isinstance(target, list):
                raise RuntimeError("deep-supervision output requires list targets")
            n = min(len(semantic), len(coarse), len(slots), len(target), len(self._ds_loss_weights))
            if main_only:
                n = min(n, 1)
            total = semantic[0].sum() * 0.0
            for index in range(n):
                weight = 1.0 if main_only else float(self._ds_loss_weights[index])
                if weight == 0.0:
                    continue
                total = total + weight * self.loss(
                    semantic[index],
                    coarse[index],
                    slots[index],
                    target[index],
                    ignore_label=self.label_manager.ignore_label,
                )
            return total

        target_main = target[0] if isinstance(target, list) else target
        return self.loss(
            semantic,
            coarse,
            slots,
            target_main,
            ignore_label=self.label_manager.ignore_label,
        )

    def on_train_start(self):
        nnUNetTrainer.on_train_start(self)
        self._setup_wandb()
        self.print_to_log_file(
            "[HierarchicalParallelAnchorSlot] "
            f"semantic_classes={NUM_ORIGINAL_CLASSES}, coarse_classes={NUM_COARSE_CHANNELS}, "
            f"groups={self.num_dynamic_groups}, slots=2, code_dim={self.hpa_code_dim}, "
            f"loss_cfg={self.hpa_loss_cfg}"
        )

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._to_device_target(batch["target"], self.device)
        self.optimizer.zero_grad(set_to_none=True)

        amp_context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
        with amp_context:
            output = self.network(data, return_hierarchy=True)
            output = self._sanitize_hierarchy(output)
            loss = self._compute_hierarchical_loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": loss.detach().cpu().numpy()}

    def on_train_epoch_end(self, train_outputs: List[dict]):
        nnUNetTrainer.on_train_epoch_end(self, train_outputs)

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._to_device_target(batch["target"], self.device)
        target_main = target[0] if isinstance(target, list) else target

        amp_context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
        with amp_context:
            output = self.network(data, return_hierarchy=True)
            output = self._sanitize_hierarchy(output)
            if self.val_loss_mode == "none":
                loss = get_main_semantic_output(output).sum() * 0.0
            else:
                loss = self._compute_hierarchical_loss(
                    output,
                    target,
                    main_only=self.val_loss_mode != "full",
                )

        logits = get_main_semantic_output(output)
        prediction = logits.argmax(dim=1)
        reference = target_main[:, 0].long()
        valid = (reference >= 0) & (reference < NUM_ORIGINAL_CLASSES)
        if self.label_manager.ignore_label is not None:
            valid &= reference != int(self.label_manager.ignore_label)

        tp = np.zeros(NUM_ORIGINAL_CLASSES - 1, dtype=np.float64)
        fp = np.zeros_like(tp)
        fn = np.zeros_like(tp)
        for label in range(1, NUM_ORIGINAL_CLASSES):
            pred_label = (prediction == label) & valid
            ref_label = (reference == label) & valid
            tp[label - 1] = torch.count_nonzero(pred_label & ref_label).item()
            fp[label - 1] = torch.count_nonzero(pred_label & ~ref_label).item()
            fn[label - 1] = torch.count_nonzero(~pred_label & ref_label).item()
        return {"loss": loss.detach().cpu().numpy(), "tp": tp, "fp": fp, "fn": fn}

    @staticmethod
    def _ddp_sum(array_value: np.ndarray) -> np.ndarray:
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, array_value)
        return np.stack(gathered, axis=0).sum(axis=0)

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs = collate_outputs(val_outputs)
        tp = np.sum(outputs["tp"], axis=0)
        fp = np.sum(outputs["fp"], axis=0)
        fn = np.sum(outputs["fn"], axis=0)
        if self.is_ddp:
            tp, fp, fn = self._ddp_sum(tp), self._ddp_sum(fp), self._ddp_sum(fn)
            gathered_losses = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_losses, outputs["loss"])
            loss_here = float(np.vstack(gathered_losses).mean())
        else:
            loss_here = float(np.mean(outputs["loss"]))

        denominator = 2.0 * tp + fp + fn
        dice = np.divide(
            2.0 * tp,
            denominator,
            out=np.full_like(denominator, np.nan, dtype=np.float64),
            where=denominator > 0,
        )
        mean_dice = float(np.nanmean(dice)) if np.any(np.isfinite(dice)) else 0.0
        dice_for_log = np.nan_to_num(dice, nan=0.0).tolist()
        report = {
            "summary": {"mean_original31_dice": mean_dice},
            "original31_dice": dice_for_log,
        }
        self._latest_structured_val_report = report
        self.logger.log("mean_fg_dice", mean_dice, self.current_epoch)
        self.logger.log("dice_per_class_or_region", dice_for_log, self.current_epoch)
        self.logger.log("val_losses", loss_here, self.current_epoch)
        self.print_to_log_file("[HierarchicalParallelAnchorSlot][val] " + json.dumps(report["summary"]))

    @torch.no_grad()
    def infer_hierarchy(self, image: torch.Tensor, use_amp: bool = True):
        self.network.eval()
        return predict_hierarchical_parallel(
            self.network,
            image,
            use_amp=use_amp,
            return_hierarchy=True,
        )

    @torch.no_grad()
    def infer_original_labels(self, image: torch.Tensor, use_amp: bool = True):
        self.network.eval()
        return predict_original_labels_hierarchical_parallel(self.network, image, use_amp=use_amp)
