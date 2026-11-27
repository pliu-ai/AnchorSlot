"""Matched-backbone CellMap adaptations of SALT, blob loss, DoDNet and HSSN.

All models expose original-label 32-channel inference. No AnchorSlot FiLM,
conditional-focus, suppression, or GroupHead auxiliary objective is inherited.
Architecture and method-loss defaults are fixed by trainer class, so checkpoint
inference does not depend on the launch shell's environment.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

from .networks import HierarchyUNet
from .hierarchy import graph_zero


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def supervision_weights(count):
    weights = np.array([2. ** -i for i in range(count)], dtype=np.float64)
    if count > 1:
        weights[-1] = 0.
    return weights / weights.sum()


def confusion_counts(scores, target, ignore_label=None):
    """Confusion without a 32-channel one-hot allocation or target mutation."""
    if target.ndim == scores.ndim:
        target = target[:, 0]
    target = target.long()
    valid = target >= 0
    if ignore_label is not None:
        valid &= target != ignore_label
    if torch.any(valid & (target >= 32)):
        raise ValueError("Invalid CellMap target label")
    pred = scores.argmax(1)
    matrix = torch.bincount(32 * target[valid] + pred[valid], minlength=32 * 32).reshape(32, 32)
    tp = matrix.diag()
    return [(x[1:].detach().cpu().numpy()) for x in (tp, matrix.sum(0) - tp, matrix.sum(1) - tp)]


class nnUNetTrainerComparisonBase(nnUNetTrainer):
    method = "flat"
    task_supervision = "sampled"

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device("cuda")):
        self._comparison_batch = int(os.environ.get("NNUNET_COMPARISON_BATCH_SIZE", "4"))
        if self._comparison_batch < 1:
            raise ValueError("NNUNET_COMPARISON_BATCH_SIZE must be positive")
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        if self.label_manager.has_regions or self.label_manager.num_segmentation_heads != 32:
            raise ValueError("These baselines require the original 32-class CellMap label map")
        if len(self.configuration_manager.patch_size) != 3 or self.is_cascaded:
            raise ValueError("Only non-cascaded 3D CellMap plans are supported")
        if os.environ.get("NNUNET_STRUCTCOND_GROUPING_CONFIG_PATH", "").strip():
            raise ValueError("Comparison baselines use the fixed published CellMap grouping")
        self.initial_lr = float(os.environ.get("NNUNET_COMPARISON_LR", ".001"))
        self.num_epochs = int(os.environ.get("NNUNET_COMPARISON_EPOCHS", "3000"))
        self.num_iterations_per_epoch = int(os.environ.get("NNUNET_COMPARISON_TRAIN_ITERS", "250"))
        self.num_val_iterations_per_epoch = int(os.environ.get("NNUNET_COMPARISON_VAL_ITERS", "25"))
        if self.initial_lr <= 0 or min(self.num_epochs, self.num_iterations_per_epoch,
                                       self.num_val_iterations_per_epoch) < 1:
            raise ValueError("Learning rate, epochs and iteration counts must be positive")
        if self.configuration_manager.batch_dice:
            raise ValueError("Matched Dataset205 comparisons require batch_dice=False")
        self.task_rng = np.random.default_rng(1234 + self.local_rank)

    def _set_batch_size_and_oversample(self):
        original = self.configuration_manager.configuration["batch_size"]
        self.configuration_manager.configuration["batch_size"] = self._comparison_batch
        try:
            super()._set_batch_size_and_oversample()
        finally:
            self.configuration_manager.configuration["batch_size"] = original

    def _do_i_compile(self):
        # Python component extraction and bounded triplet mining are intentional.
        return False

    @classmethod
    def build_network_architecture(cls, architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        if num_output_channels != 32:
            raise ValueError("Expected 32 original-label output channels")
        raw_channels = {"salt": 45, "hssn": 50}.get(cls.method, 32)
        backbone = get_network_from_plans(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels, raw_channels, allow_init=True,
            deep_supervision=enable_deep_supervision)
        if cls.method in {"salt", "hssn"}:
            return HierarchyUNet(backbone, cls.method)
        if cls.method == "dodnet":
            from .dodnet import DoDNetUNet
            return DoDNetUNet(backbone)
        return backbone

    def initialize(self):
        if self.was_initialized:
            raise RuntimeError("initialize() called twice")
        self.num_input_channels = determine_num_input_channels(
            self.plans_manager, self.configuration_manager, self.dataset_json)
        self.network = self.build_network_architecture(
            self.configuration_manager.network_arch_class_name,
            self.configuration_manager.network_arch_init_kwargs,
            self.configuration_manager.network_arch_init_kwargs_req_import,
            self.num_input_channels, 32, self.enable_deep_supervision).to(self.device)
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        if self.is_ddp:
            self.network = DDP(self.network,
                               device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                               find_unused_parameters=False)
        self.loss = self._build_loss()
        self.was_initialized = True

    def _build_loss(self):
        ignore = self.label_manager.ignore_label
        if self.method == "salt":
            from .salt import SALTLoss
            return SALTLoss(ignore_label=ignore)
        if self.method == "hssn":
            from .hssn import HSSNLoss
            return HSSNLoss(ignore_label=ignore)
        if self.method == "dodnet":
            from .task_loss import binary_task_loss
            return binary_task_loss
        from .blob_loss import FlatCEAndDiceAndBlobLoss
        return FlatCEAndDiceAndBlobLoss(ignore_label=ignore,
                                       weight_blob=.5 if self.method == "blob" else 0.)

    def _prepare(self, batch, sample_tasks=True):
        target_cpu = _as_list(batch["target"])
        components = None
        if self.method == "blob":
            from .blob_loss import prepare_blob_components
            components = [prepare_blob_components(t, 32, ignore_label=self.label_manager.ignore_label,
                                                  spatial_dims=3) if weight else None
                          for t, weight in zip(target_cpu, supervision_weights(len(target_cpu)))]
        targets = [t.to(self.device, non_blocking=True) for t in target_cpu]
        data = batch["data"].to(self.device, non_blocking=True)
        ids = None
        if self.method == "dodnet" and sample_tasks and self.task_supervision == "sampled":
            from .task_loss import sample_task_ids
            ids = torch.as_tensor(sample_task_ids(target_cpu[0], self.task_rng), device=self.device)
        return data, targets, ids, components

    def _forward_loss(self, data, targets, ids=None, validation=False, components=None):
        features = None
        all_tasks = self.method == "dodnet" and (validation or self.task_supervision == "all")
        if self.method in {"salt", "hssn"}:
            result = self.network(data, return_aux=True)
            outputs, features = _as_list(result["raw"]), result["features"]
        else:
            result = self.network(data, None if all_tasks else ids) if self.method == "dodnet" else self.network(data)
            outputs = _as_list(result)
        if len(outputs) != len(targets):
            raise ValueError("Deep-supervision output/target count mismatch")
        total = graph_zero(outputs[0])
        progress = self.current_epoch / max(1, self.num_epochs - 1)
        for i, (output, target, weight) in enumerate(zip(outputs, targets, supervision_weights(len(outputs)))):
            if weight == 0:
                # Retain autograd edges for unused DS heads, with exactly zero
                # objective weight, so DDP works without find_unused_parameters.
                total = total + graph_zero(output)
                continue
            if self.method == "hssn":
                # Add the feature objective only once, without scaling it by DS weight.
                loss = self.loss(output, target, features=None, progress=progress)
            elif all_tasks:
                from .dodnet import DEFAULT_DODNET_TASK_LABELS
                losses = []
                for task, labels in enumerate(DEFAULT_DODNET_TASK_LABELS):
                    local = torch.cat([output[:, list(labels)]] +
                                      [torch.zeros_like(output[:, :1])] * (3 - len(labels)), dim=1)
                    task_ids = torch.full((data.shape[0],), task, device=data.device)
                    losses.append(self.loss(local, target, task_ids, self.label_manager.ignore_label))
                loss = torch.stack(losses).mean()
            elif self.method == "dodnet":
                loss = self.loss(output, target, ids, self.label_manager.ignore_label)
            elif self.method == "blob":
                loss = self.loss(output, target, components=components[i] if components else None)
            else:
                loss = self.loss(output, target)
            total = total + float(weight) * loss
        if self.method == "hssn":
            # HSSNLoss exposes its triplet term separately; no duplicated FTM.
            total = total + self.loss.feature_loss(features, targets[0], progress=progress)
        if validation:
            if self.method in {"salt", "hssn"}:
                module = self.network.module if self.is_ddp else self.network
                scores = module.leaf_scores(outputs[0].float())
            else:
                scores = outputs[0].float()
            return total, scores
        return total

    def train_step(self, batch):
        data, targets, ids, components = self._prepare(batch)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            loss = self._forward_loss(data, targets, ids, components=components)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {self.method} training loss")
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12, error_if_nonfinite=True)
            self.optimizer.step()
        return {"loss": loss.detach().cpu().numpy()}

    @torch.no_grad()
    def validation_step(self, batch):
        data, targets, _, components = self._prepare(batch, sample_tasks=False)
        with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            loss, scores = self._forward_loss(data, targets, validation=True, components=components)
        tp, fp, fn = confusion_counts(scores, targets[0], self.label_manager.ignore_label)
        return {"loss": loss.detach().cpu().numpy(), "tp_hard": tp, "fp_hard": fp, "fn_hard": fn}

    def comparison_recipe(self):
        return {"schema": 1, "method": self.method, "trainer": type(self).__name__,
                "global_batch_size": self._comparison_batch, "initial_lr": self.initial_lr,
                "weight_decay": self.weight_decay, "epochs": self.num_epochs,
                "train_iters": self.num_iterations_per_epoch, "val_iters": self.num_val_iterations_per_epoch,
                "label_contract": "CellMap original IDs 0..31; fixed 13 dynamic groups",
                "deep_supervision": "normalized 2^-i; coarsest zero (autograd retained)",
                "sampler": "nnUNet spatial sampler; task supervision specified separately",
                "task_supervision": self.task_supervision if self.method == "dodnet" else "all_labels",
                "method_defaults": "documentation/comparison_baselines.md"}

    def on_train_start(self):
        super().on_train_start()
        if self.local_rank == 0:
            Path(self.output_folder, "comparison_recipe.json").write_text(
                json.dumps(self.comparison_recipe(), indent=2) + "\n")

    def save_checkpoint(self, filename):
        if self.local_rank != 0 or self.disable_checkpointing:
            return
        module = self.network.module if self.is_ddp else self.network
        torch.save({"network_weights": module.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "grad_scaler_state": self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    "logging": self.logger.get_checkpoint(), "_best_ema": self._best_ema,
                    "current_epoch": self.current_epoch + 1, "init_args": self.my_init_kwargs,
                    "trainer_name": type(self).__name__,
                    "inference_allowed_mirroring_axes": self.inference_allowed_mirroring_axes,
                    "comparison_recipe": self.comparison_recipe(),
                    "task_rng_state": self.task_rng.bit_generator.state}, filename)

    def load_checkpoint(self, filename_or_checkpoint):
        checkpoint = (torch.load(filename_or_checkpoint, map_location="cpu", weights_only=False)
                      if isinstance(filename_or_checkpoint, (str, Path)) else filename_or_checkpoint)
        if checkpoint.get("comparison_recipe") != self.comparison_recipe():
            raise ValueError("Checkpoint comparison recipe differs; refusing an incompatible resume")
        super().load_checkpoint(checkpoint)
        # Stored rank-0 state plus disjoint rank streams: statistically equivalent,
        # not a promise of bitwise continuation of asynchronous data augmentation.
        self.task_rng.bit_generator.state = checkpoint["task_rng_state"]
        if self.local_rank:
            self.task_rng = np.random.Generator(self.task_rng.bit_generator.jumped(self.local_rank))


class nnUNetTrainerCellMapFlatControl(nnUNetTrainerComparisonBase):
    method = "flat"


class nnUNetTrainerCellMapSALT(nnUNetTrainerComparisonBase):
    method = "salt"


class nnUNetTrainerCellMapBlobLoss(nnUNetTrainerComparisonBase):
    method = "blob"


class nnUNetTrainerCellMapDoDNet(nnUNetTrainerComparisonBase):
    method = "dodnet"


class nnUNetTrainerCellMapDoDNetAllTasks(nnUNetTrainerCellMapDoDNet):
    """Control for supervision quantity: every task is trained on every patch."""
    task_supervision = "all"


class nnUNetTrainerCellMapHSSN(nnUNetTrainerComparisonBase):
    method = "hssn"


TRAINERS = {cls.method: cls for cls in (
    nnUNetTrainerCellMapFlatControl, nnUNetTrainerCellMapSALT,
    nnUNetTrainerCellMapBlobLoss, nnUNetTrainerCellMapDoDNet, nnUNetTrainerCellMapHSSN)}
TRAINERS["dodnet_all"] = nnUNetTrainerCellMapDoDNetAllTasks
