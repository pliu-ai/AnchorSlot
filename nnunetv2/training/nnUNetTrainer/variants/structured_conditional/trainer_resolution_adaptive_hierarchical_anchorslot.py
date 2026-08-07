from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
from torch import autocast

from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from .hierarchical_parallel_mapping import NUM_COARSE_CHANNELS, NUM_ORIGINAL_CLASSES
from .network_hierarchical_parallel_anchorslot import get_main_semantic_output
from .network_resolution_adaptive_hierarchical_anchorslot import (
    ResolutionAdaptiveHierarchicalAnchorSlotUNet,
)
from .resolution_adaptive_mapping import (
    ATOMIC_NAMES,
    NUM_PARENT_CLASSES,
    PARENT_NAMES,
)
from .structured_loss_resolution_adaptive_hierarchical_anchorslot import (
    ResolutionAdaptiveHierarchicalAnchorSlotLoss,
    ResolutionAdaptiveLossConfig,
)
from .trainer_hierarchical_parallel_anchorslot import (
    nnUNetTrainerHierarchicalParallelAnchorSlot,
)


class _MixedResolutionIterator:
    """Mix two already-augmented nnU-Net streams and attach physical spacing."""

    def __init__(
        self,
        primary,
        auxiliary,
        primary_spacing: Sequence[float],
        auxiliary_spacing: Sequence[float],
        auxiliary_probability: float,
        *,
        random_sampling: bool,
        seed: int,
    ) -> None:
        self.primary = primary
        self.auxiliary = auxiliary
        self.primary_spacing = tuple(float(value) for value in primary_spacing)
        self.auxiliary_spacing = tuple(float(value) for value in auxiliary_spacing)
        self.auxiliary_probability = float(auxiliary_probability)
        self.random_sampling = random_sampling
        self.rng = np.random.RandomState(seed)
        self._next_auxiliary = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.random_sampling:
            use_auxiliary = bool(self.rng.random_sample() < self.auxiliary_probability)
        else:
            use_auxiliary = self._next_auxiliary
            self._next_auxiliary = not self._next_auxiliary
        source = self.auxiliary if use_auxiliary else self.primary
        spacing = self.auxiliary_spacing if use_auxiliary else self.primary_spacing
        batch = dict(next(source))
        batch_size = int(batch["data"].shape[0])
        batch["voxel_size"] = torch.tensor(spacing, dtype=torch.float32)[None].expand(batch_size, -1)
        batch["resolution_source"] = "auxiliary" if use_auxiliary else "primary"
        return batch

    def finish_children(self) -> None:
        for child in (self.primary, self.auxiliary):
            finish = getattr(child, "_finish", None)
            if finish is not None:
                finish()


class nnUNetTrainerResolutionAdaptiveHierarchicalParallelAnchorSlot(
    nnUNetTrainerHierarchicalParallelAnchorSlot
):
    """Resolution-aware, annotation-aware HPA trainer for high/low CellMap data."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.native_labels_root = os.environ.get("NNUNET_RAHPA_NATIVE_LABELS_ROOT")
        self._annotation_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.ra_loss_cfg = ResolutionAdaptiveLossConfig(
            lambda_semantic_ce=float(os.environ.get("NNUNET_HPA_LAMBDA_SEMANTIC_CE", "1.0")),
            lambda_semantic_dice=float(os.environ.get("NNUNET_HPA_LAMBDA_SEMANTIC_DICE", "1.0")),
            lambda_coarse_ce=float(os.environ.get("NNUNET_HPA_LAMBDA_COARSE_CE", "0.5")),
            lambda_slot_ce=float(os.environ.get("NNUNET_HPA_LAMBDA_SLOT_CE", "0.5")),
            lambda_parent_bce=float(os.environ.get("NNUNET_RAHPA_LAMBDA_PARENT", "0.5")),
            lambda_hierarchy_consistency=float(os.environ.get("NNUNET_RAHPA_LAMBDA_HIERARCHY", "0.25")),
            lambda_boundary=float(os.environ.get("NNUNET_RAHPA_LAMBDA_BOUNDARY", "0.1")),
            lambda_affinity=float(os.environ.get("NNUNET_RAHPA_LAMBDA_AFFINITY", "0.1")),
            ignore_unannotated_background=os.environ.get(
                "NNUNET_RAHPA_IGNORE_UNANNOTATED_BACKGROUND", "0"
            ).lower() in {"1", "true", "yes"},
        )
        self.aux_dataset_id = os.environ.get("NNUNET_RAHPA_AUX_DATASET_ID")
        self.aux_configuration_name = os.environ.get(
            "NNUNET_RAHPA_AUX_CONFIGURATION", "3d_fullres"
        )
        self.aux_probability = float(
            os.environ.get("NNUNET_RAHPA_AUX_PROBABILITY", "0.5")
        )
        if not 0.0 <= self.aux_probability <= 1.0:
            raise ValueError("NNUNET_RAHPA_AUX_PROBABILITY must be in [0, 1]")
        self.num_iterations_per_epoch = int(
            os.environ.get(
                "NNUNET_RAHPA_NUM_TRAIN_ITERS_PER_EPOCH",
                str(self.num_iterations_per_epoch),
            )
        )
        if self.num_iterations_per_epoch < 1:
            raise ValueError("NNUNET_RAHPA_NUM_TRAIN_ITERS_PER_EPOCH must be >= 1")

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
        reference = tuple(
            float(value)
            for value in os.environ.get("NNUNET_RAHPA_REFERENCE_VOXEL_SIZE", "4,4,4").split(",")
        )
        return ResolutionAdaptiveHierarchicalAnchorSlotUNet(
            backbone=backbone,
            code_dim=code_dim,
            reference_voxel_size=reference,
        )

    def _build_loss(self):
        return ResolutionAdaptiveHierarchicalAnchorSlotLoss(self.ra_loss_cfg)

    def _auxiliary_dataloaders(self):
        dataset_id = int(self.aux_dataset_id)
        preprocessed_root = Path(self.preprocessed_dataset_folder_base).parent
        matches = sorted(preprocessed_root.glob(f"Dataset{dataset_id:03d}_*"))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one preprocessed Dataset{dataset_id:03d}_* directory, got {matches}"
            )
        dataset_dir = matches[0]
        plans_path = dataset_dir / f"{self.plans_manager.plans_name}.json"
        dataset_json_path = dataset_dir / "dataset.json"
        if not plans_path.is_file() or not dataset_json_path.is_file():
            raise FileNotFoundError(
                f"Auxiliary dataset needs {plans_path.name} and dataset.json under {dataset_dir}"
            )
        aux_plans = PlansManager(load_json(str(plans_path)))
        aux_configuration = aux_plans.get_configuration(self.aux_configuration_name)
        aux_dataset_json = load_json(str(dataset_json_path))
        if len(aux_configuration.patch_size) != len(self.configuration_manager.patch_size):
            raise ValueError("Primary and auxiliary configurations must have the same dimensionality")

        primary_splits_path = Path(self.preprocessed_dataset_folder_base) / "splits_final.json"
        auxiliary_splits_path = dataset_dir / "splits_final.json"
        if primary_splits_path.is_file() and auxiliary_splits_path.is_file():
            primary_splits = load_json(str(primary_splits_path))
            auxiliary_splits = load_json(str(auxiliary_splits_path))
            if self.fold < min(len(primary_splits), len(auxiliary_splits)):
                primary_train = set(primary_splits[self.fold]["train"])
                primary_val = set(primary_splits[self.fold]["val"])
                auxiliary_train = set(auxiliary_splits[self.fold]["train"])
                auxiliary_val = set(auxiliary_splits[self.fold]["val"])
                leakage = (primary_val & auxiliary_train) | (auxiliary_val & primary_train)
                if leakage:
                    raise RuntimeError(
                        "Cross-resolution train/validation leakage detected: "
                        f"{sorted(leakage)}"
                    )

        state_names = (
            "plans_manager", "configuration_manager", "configuration_name",
            "dataset_json", "label_manager", "preprocessed_dataset_folder_base",
            "preprocessed_dataset_folder", "is_cascaded", "folder_with_segs_from_previous_stage",
        )
        saved = {name: getattr(self, name) for name in state_names}
        try:
            self.plans_manager = aux_plans
            self.configuration_manager = aux_configuration
            self.configuration_name = self.aux_configuration_name
            self.dataset_json = aux_dataset_json
            self.label_manager = aux_plans.get_label_manager(aux_dataset_json)
            self.preprocessed_dataset_folder_base = str(dataset_dir)
            self.preprocessed_dataset_folder = str(
                dataset_dir / aux_configuration.data_identifier
            )
            self.is_cascaded = False
            self.folder_with_segs_from_previous_stage = None
            loaders = nnUNetTrainer.get_dataloaders(self)
        finally:
            for name, value in saved.items():
                setattr(self, name, value)
        return (*loaders, tuple(aux_configuration.spacing), str(dataset_dir.name))

    def get_dataloaders(self):
        primary_train, primary_val = nnUNetTrainer.get_dataloaders(self)
        if self.aux_dataset_id is None:
            return primary_train, primary_val
        primary_spacing = tuple(self.configuration_manager.spacing)
        aux_train, aux_val, aux_spacing, aux_name = self._auxiliary_dataloaders()
        seed = 12345 + int(getattr(self, "local_rank", 0))
        self.print_to_log_file(
            "[ResolutionAdaptiveHPA] mixed-resolution loader: "
            f"primary={self.plans_manager.dataset_name}@{primary_spacing}, "
            f"auxiliary={aux_name}@{aux_spacing}, p_aux={self.aux_probability}"
        )
        return (
            _MixedResolutionIterator(
                primary_train, aux_train, primary_spacing, aux_spacing,
                self.aux_probability, random_sampling=True, seed=seed,
            ),
            _MixedResolutionIterator(
                primary_val, aux_val, primary_spacing, aux_spacing,
                0.5, random_sampling=False, seed=seed + 1,
            ),
        )

    def _case_annotation_masks(self, case_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        cached = self._annotation_cache.get(case_key)
        if cached is not None:
            return cached
        atomic = torch.ones(NUM_ORIGINAL_CLASSES, dtype=torch.bool)
        parents = torch.ones(NUM_PARENT_CLASSES, dtype=torch.bool)
        if self.native_labels_root:
            atomic.zero_()
            atomic[0] = True
            parents.zero_()
            try:
                dataset, crop_id = case_key.rsplit("_crop", 1)
                crop_name = f"crop{int(crop_id)}"
                label_dir = Path(self.native_labels_root) / dataset / crop_name / "labels"
                if not label_dir.is_dir():
                    raise FileNotFoundError(label_dir)
                prefix = f"{dataset}_{crop_name}_"
                active_names = set()
                known_names = sorted((*ATOMIC_NAMES, *PARENT_NAMES), key=len, reverse=True)
                for path in label_dir.glob("*.nii.gz"):
                    if not path.name.startswith(prefix):
                        continue
                    suffix = path.name[len(prefix) :]
                    name = next(
                        (candidate for candidate in known_names if suffix.startswith(f"{candidate}_")),
                        None,
                    )
                    if name is not None:
                        active_names.add(name)
                for index, name in enumerate(ATOMIC_NAMES, start=1):
                    atomic[index] = name in active_names
                for index, name in enumerate(PARENT_NAMES):
                    parents[index] = name in active_names
            except (ValueError, FileNotFoundError):
                # A missing sidecar must never turn the entire batch into a
                # zero-loss batch. Full supervision is the backward-compatible fallback.
                atomic.fill_(True)
                parents.fill_(True)
        self._annotation_cache[case_key] = (atomic, parents)
        return atomic, parents

    def _batch_annotation_masks(self, batch: dict, target) -> Tuple[torch.Tensor, torch.Tensor]:
        keys = [str(key) for key in batch.get("keys", [])]
        target_main = target[0] if isinstance(target, list) else target
        if not keys:
            keys = [""] * int(target_main.shape[0])
        atomic_masks, parent_masks = zip(
            *(self._case_annotation_masks(key) for key in keys)
        )
        atomic = torch.stack(atomic_masks).to(self.device)
        parents = torch.stack(parent_masks).to(self.device)
        return atomic, parents

    def _batch_voxel_size(self, batch: dict, data: torch.Tensor) -> torch.Tensor:
        spatial_dims = data.ndim - 2
        values = batch.get("voxel_size")
        if values is None:
            spacing = tuple(float(value) for value in self.configuration_manager.spacing)
            if len(spacing) != spatial_dims:
                spacing = spacing[-spatial_dims:]
            values = torch.tensor(spacing, dtype=torch.float32)[None].expand(data.shape[0], -1)
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def _compute_resolution_adaptive_loss(
        self,
        output: dict,
        target,
        active_semantic: torch.Tensor,
        active_parent: torch.Tensor,
        main_only: bool = False,
    ) -> torch.Tensor:
        keys = ("semantic_logits", "coarse_logits", "slot_logits", "parent_logits", "separation_logits")
        values = [output[key] for key in keys]
        if isinstance(values[0], (tuple, list)):
            if not isinstance(target, list):
                raise RuntimeError("deep-supervision output requires list targets")
            n = min(*(len(value) for value in values), len(target), len(self._ds_loss_weights))
            if main_only:
                n = min(n, 1)
            total = values[0][0].sum() * 0.0
            for index in range(n):
                weight = 1.0 if main_only else float(self._ds_loss_weights[index])
                if weight == 0.0:
                    continue
                total = total + weight * self.loss(
                    *(value[index] for value in values),
                    target[index],
                    ignore_label=self.label_manager.ignore_label,
                    active_semantic_mask=active_semantic,
                    active_parent_annotation_mask=active_parent,
                )
            return total
        target_main = target[0] if isinstance(target, list) else target
        return self.loss(
            *values,
            target_main,
            ignore_label=self.label_manager.ignore_label,
            active_semantic_mask=active_semantic,
            active_parent_annotation_mask=active_parent,
        )

    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file(
            "[ResolutionAdaptiveHPA] "
            f"parent_classes={NUM_PARENT_CLASSES}, loss_cfg={self.ra_loss_cfg}, "
            f"native_labels_root={self.native_labels_root!r}, "
            f"spacing={tuple(self.configuration_manager.spacing)}"
        )

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._to_device_target(batch["target"], self.device)
        voxel_size = self._batch_voxel_size(batch, data)
        active_semantic, active_parent = self._batch_annotation_masks(batch, target)
        self.optimizer.zero_grad(set_to_none=True)
        amp_context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
        with amp_context:
            output = self.network(data, return_hierarchy=True, voxel_size=voxel_size)
            output = self._sanitize_hierarchy(output)
            loss = self._compute_resolution_adaptive_loss(
                output, target, active_semantic, active_parent
            )
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

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._to_device_target(batch["target"], self.device)
        target_main = target[0] if isinstance(target, list) else target
        voxel_size = self._batch_voxel_size(batch, data)
        active_semantic, active_parent = self._batch_annotation_masks(batch, target)
        amp_context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
        with amp_context:
            output = self.network(data, return_hierarchy=True, voxel_size=voxel_size)
            output = self._sanitize_hierarchy(output)
            loss = self._compute_resolution_adaptive_loss(
                output, target, active_semantic, active_parent,
                main_only=self.val_loss_mode != "full",
            ) if self.val_loss_mode != "none" else get_main_semantic_output(output).sum() * 0.0

        prediction = get_main_semantic_output(output).argmax(dim=1)
        reference = target_main[:, 0].long()
        valid = (reference >= 0) & (reference < NUM_ORIGINAL_CLASSES)
        if self.label_manager.ignore_label is not None:
            valid &= reference != int(self.label_manager.ignore_label)
        tp = np.zeros(NUM_ORIGINAL_CLASSES - 1, dtype=np.float64)
        fp = np.zeros_like(tp)
        fn = np.zeros_like(tp)
        for label in range(1, NUM_ORIGINAL_CLASSES):
            batch_active = active_semantic[:, label].view(-1, *([1] * (reference.ndim - 1)))
            label_valid = valid & batch_active
            pred_label = (prediction == label) & label_valid
            ref_label = (reference == label) & label_valid
            tp[label - 1] = torch.count_nonzero(pred_label & ref_label).item()
            fp[label - 1] = torch.count_nonzero(pred_label & ~ref_label).item()
            fn[label - 1] = torch.count_nonzero(~pred_label & ref_label).item()
        return {
            "loss": loss.detach().cpu().numpy(),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "resolution_nm": float(voxel_size.mean().detach().cpu()),
        }

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        super().on_validation_epoch_end(val_outputs)
        by_resolution = {}
        for resolution in sorted({float(output["resolution_nm"]) for output in val_outputs}):
            selected = [
                output for output in val_outputs
                if np.isclose(float(output["resolution_nm"]), resolution)
            ]
            tp = np.sum([output["tp"] for output in selected], axis=0)
            fp = np.sum([output["fp"] for output in selected], axis=0)
            fn = np.sum([output["fn"] for output in selected], axis=0)
            if self.is_ddp:
                tp, fp, fn = self._ddp_sum(tp), self._ddp_sum(fp), self._ddp_sum(fn)
            denominator = 2.0 * tp + fp + fn
            dice = np.divide(
                2.0 * tp, denominator,
                out=np.full_like(denominator, np.nan, dtype=np.float64),
                where=denominator > 0,
            )
            mean_dice = float(np.nanmean(dice)) if np.any(np.isfinite(dice)) else 0.0
            key = f"{resolution:g}nm"
            by_resolution[key] = {
                "mean_original31_dice": mean_dice,
                "original31_dice": np.nan_to_num(dice, nan=0.0).tolist(),
                "num_batches": len(selected),
            }
            self.logger.log(f"mean_fg_dice_{key}", mean_dice, self.current_epoch)
        self._latest_structured_val_report["by_resolution"] = by_resolution
        if self.local_rank == 0:
            save_json(
                {
                    "epoch": int(self.current_epoch),
                    "overall": self._latest_structured_val_report.get("summary", {}),
                    "by_resolution": by_resolution,
                },
                str(Path(self.output_folder) / "latest_resolution_validation.json"),
                sort_keys=False,
            )
        self.print_to_log_file(f"[ResolutionAdaptiveHPA][val-by-resolution] {by_resolution}")

    def on_train_end(self):
        train_loader, val_loader = self.dataloader_train, self.dataloader_val
        super().on_train_end()
        for loader in (train_loader, val_loader):
            if isinstance(loader, _MixedResolutionIterator):
                loader.finish_children()

    @torch.no_grad()
    def infer_hierarchy(
        self,
        image: torch.Tensor,
        voxel_size: Sequence[float] | torch.Tensor | None = None,
        use_amp: bool = True,
    ):
        self.network.eval()
        amp_context = autocast(image.device.type, enabled=use_amp) if image.device.type == "cuda" else dummy_context()
        with amp_context:
            return self.network(
                image, return_hierarchy=True,
                voxel_size=voxel_size if voxel_size is not None else self.configuration_manager.spacing,
            )


# Kept as a source-level alias for early development checkpoints.
nnUNetTrainerResolutionAdaptiveHierarchicalAnchorSlot = (
    nnUNetTrainerResolutionAdaptiveHierarchicalParallelAnchorSlot
)
