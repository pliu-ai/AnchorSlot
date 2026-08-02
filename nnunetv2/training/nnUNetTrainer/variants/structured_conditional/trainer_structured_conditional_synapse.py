from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from torch import autocast
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

from .dataloader_structured_conditional_synapse import StructuredConditionalDataLoader3D
from .inference_structured_conditional_synapse import (
    predict_logits_all_groups,
    predict_logits_for_group,
    reconstruct_original_labels_from_all_groups,
)
from .label_mapping_synapse import (
    COND_SLOT_1_CHANNEL,
    DYNAMIC_GROUP_SPECS,
    NUM_DYNAMIC_GROUPS,
    NUM_OUTPUT_CHANNELS,
    infer_present_groups_from_segmentation,
    remap_original_to_structured,
    sample_group_id_for_case,
)
from .condition_encoding import build_group_text_matrix, build_group_text_matrix_from_group_embeddings
from .metrics_structured_conditional_synapse import (
    build_validation_report,
    compute_group_confusion_from_logits,
    empty_validation_accumulators,
)
from .network_structured_conditional import StructuredConditionalUNet, get_main_output
from .structured_loss_synapse import StructuredConditionalLoss, StructuredLossConfig


def _load_text_embedding_blob(text_emb_path: str) -> dict:
    if not text_emb_path:
        raise ValueError("text_emb_path is required")
    return torch.load(text_emb_path, map_location="cpu", weights_only=False)


def _infer_text_embedding_dim(text_emb_path: str) -> int:
    blob = _load_text_embedding_blob(text_emb_path)
    if "embedding_dim" in blob:
        return int(blob["embedding_dim"])
    for dict_key in ("label_embeddings", "anchor_embeddings", "group_embeddings"):
        values = blob.get(dict_key, {})
        if isinstance(values, dict):
            for value in values.values():
                if torch.is_tensor(value):
                    return int(value.reshape(-1).shape[0])
    raise ValueError(f"Could not infer text embedding dim from {text_emb_path}")


def _build_group_slot_text_embedding_matrix(
    text_emb_path: str,
    num_slots: int = 3,
) -> Tuple[torch.Tensor, Dict[int, Tuple[str, ...]], List[str]]:
    blob = _load_text_embedding_blob(text_emb_path)
    label_embeddings = blob.get("label_embeddings", {})
    if not isinstance(label_embeddings, dict):
        raise KeyError(f"{text_emb_path} must contain a dict key 'label_embeddings'")

    text_dim = _infer_text_embedding_dim(text_emb_path)
    matrix = torch.zeros((NUM_DYNAMIC_GROUPS, int(num_slots), text_dim), dtype=torch.float32)
    group_to_labels: Dict[int, Tuple[str, ...]] = {}
    missing: List[str] = []

    for spec in DYNAMIC_GROUP_SPECS:
        labels = tuple(str(i) for i in spec.subclass_names)
        if len(labels) > int(num_slots):
            raise ValueError(f"{spec.short_name} has {len(labels)} labels but num_slots={num_slots}")
        group_to_labels[int(spec.group_id)] = labels
        for slot_idx, label_name in enumerate(labels):
            value = label_embeddings.get(label_name)
            if value is None:
                missing.append(label_name)
                continue
            matrix[int(spec.group_id), int(slot_idx)] = torch.nn.functional.normalize(
                value.detach().float().reshape(-1),
                dim=-1,
            )

    missing = sorted(set(missing))
    if missing:
        raise KeyError(
            "Missing BioMedCLIP label_embeddings for text-generated dynamic slot labels: "
            + json.dumps(missing)
        )
    return matrix, group_to_labels, missing


def _build_group_text_matrix_from_label_embeddings(text_emb_path: str) -> torch.Tensor:
    blob = _load_text_embedding_blob(text_emb_path)
    label_embeddings = blob.get("label_embeddings", {})
    if not isinstance(label_embeddings, dict):
        raise KeyError(f"{text_emb_path} must contain a dict key 'label_embeddings'")

    rows = []
    missing: List[str] = []
    for spec in DYNAMIC_GROUP_SPECS:
        values = []
        for label_name in spec.subclass_names:
            value = label_embeddings.get(label_name)
            if value is None:
                missing.append(str(label_name))
                continue
            values.append(value.detach().float().reshape(-1))
        if len(values) == 0:
            raise KeyError(f"No label_embeddings found for dynamic group {spec.short_name}: {spec.subclass_names}")
        rows.append(torch.nn.functional.normalize(torch.stack(values, dim=0).mean(dim=0), dim=-1))
    if missing:
        raise KeyError(
            "Missing BioMedCLIP label_embeddings for generated grouping text conditioning: "
            + json.dumps(sorted(set(missing)))
        )
    return torch.stack(rows, dim=0)


class nnUNetTrainerStructuredConditionalSynapse(nnUNetTrainer):
    """
    Structured conditional trainer for CellMap with ER in dynamic slots:
    - one shared model
    - fixed 9-channel output head
    - dynamic group-conditioned remapping
    - unified train/val/test workflow
    """

    default_amp_dtype = "float16"

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        self._batch_size_override_from_env = int(os.environ.get("NNUNET_STRUCTCOND_BATCH_SIZE", "0"))
        if self._batch_size_override_from_env < 0:
            raise ValueError("NNUNET_STRUCTCOND_BATCH_SIZE must be >= 0.")

        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # Keep one fixed head independent of dataset label count.
        self.fixed_num_output_channels = NUM_OUTPUT_CHANNELS
        self.num_dynamic_groups = NUM_DYNAMIC_GROUPS
        self.grouping_config_path = str(os.environ.get("NNUNET_STRUCTCOND_GROUPING_CONFIG_PATH", "")).strip()
        self.global_batch_size_override = self._batch_size_override_from_env

        # Group sampling during training.
        self.p_present_group = float(np.clip(float(os.environ.get("NNUNET_STRUCTCOND_P_PRESENT_GROUP", "0.8")), 0.0, 1.0))
        self.group_sampling_seed = int(os.environ.get("NNUNET_STRUCTCOND_GROUP_SEED", "1234"))
        self.group_sampling_rng = np.random.default_rng(self.group_sampling_seed)

        # Optimizer defaults: keep close to previously stable conditional training.
        self.initial_lr = float(os.environ.get("NNUNET_STRUCTCOND_INITIAL_LR", "0.001"))
        self.weight_decay = float(os.environ.get("NNUNET_STRUCTCOND_WEIGHT_DECAY", str(self.weight_decay)))
        self.num_epochs = int(os.environ.get("NNUNET_STRUCTCOND_NUM_EPOCHS", str(self.num_epochs)))
        self.num_val_iterations_per_epoch = int(
            os.environ.get("NNUNET_STRUCTCOND_NUM_VAL_ITERS_PER_EPOCH", str(self.num_val_iterations_per_epoch))
        )
        if self.initial_lr <= 0:
            raise ValueError("NNUNET_STRUCTCOND_INITIAL_LR must be > 0.")
        if self.weight_decay < 0:
            raise ValueError("NNUNET_STRUCTCOND_WEIGHT_DECAY must be >= 0.")
        if self.num_epochs < 1:
            raise ValueError("NNUNET_STRUCTCOND_NUM_EPOCHS must be >= 1.")
        if self.num_val_iterations_per_epoch < 1:
            raise ValueError("NNUNET_STRUCTCOND_NUM_VAL_ITERS_PER_EPOCH must be >= 1.")

        amp_dtype_name = str(
            os.environ.get("NNUNET_STRUCTCOND_AMP_DTYPE", self.default_amp_dtype)
        ).strip().lower()
        amp_dtype_aliases = {
            "fp16": "float16",
            "half": "float16",
            "bf16": "bfloat16",
            "fp32": "float32",
            "none": "float32",
            "off": "float32",
        }
        amp_dtype_name = amp_dtype_aliases.get(amp_dtype_name, amp_dtype_name)
        if amp_dtype_name not in {"float16", "bfloat16", "float32"}:
            raise ValueError(
                "NNUNET_STRUCTCOND_AMP_DTYPE must be one of: float16, bfloat16, float32"
            )
        if amp_dtype_name == "bfloat16" and self.device.type == "cuda":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("This GPU does not support CUDA bfloat16 training.")

        self.amp_dtype_name = amp_dtype_name
        self.amp_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": None,
        }[amp_dtype_name]
        self.loss_in_fp32 = str(
            os.environ.get("NNUNET_STRUCTCOND_LOSS_FP32", "1")
        ).lower() in {"1", "true", "yes", "y"}
        self.grad_clip_norm = float(os.environ.get("NNUNET_STRUCTCOND_GRAD_CLIP_NORM", "12"))
        self.skip_nonfinite_grad = str(
            os.environ.get("NNUNET_STRUCTCOND_SKIP_NONFINITE_GRAD", "0")
        ).lower() in {"1", "true", "yes", "y"}
        self._nonfinite_grad_skip_count = 0
        self.reset_optimizer_on_load = str(
            os.environ.get("NNUNET_STRUCTCOND_RESET_OPTIMIZER_ON_LOAD", "0")
        ).lower() in {"1", "true", "yes", "y"}
        self.freeze_conditioning_on_load = str(
            os.environ.get("NNUNET_STRUCTCOND_FREEZE_CONDITIONING_ON_LOAD", "0")
        ).lower() in {"1", "true", "yes", "y"}
        self._conditioning_optimizer_frozen = False
        if self.grad_clip_norm <= 0:
            raise ValueError("NNUNET_STRUCTCOND_GRAD_CLIP_NORM must be > 0.")

        self.use_text_generated_slot_head = (
            str(os.environ.get("NNUNET_STRUCTCOND_USE_TEXT_GENERATED_SLOT_HEAD", "0")).lower()
            in {"1", "true", "yes", "y"}
        )
        self.text_generated_slot_embedding_path = str(os.environ.get("NNUNET_STRUCTCOND_TEXT_EMB_PATH", "")).strip()
        self.text_generated_slot_hidden_dim = int(
            os.environ.get("NNUNET_STRUCTCOND_TEXT_GENERATED_SLOT_HIDDEN_DIM", "512")
        )
        self.text_generated_slot_use_bias = (
            str(os.environ.get("NNUNET_STRUCTCOND_TEXT_GENERATED_SLOT_USE_BIAS", "1")).lower()
            in {"1", "true", "yes", "y"}
        )
        self.text_generated_slot_normalize_weight = (
            str(os.environ.get("NNUNET_STRUCTCOND_TEXT_GENERATED_SLOT_NORMALIZE_WEIGHT", "1")).lower()
            in {"1", "true", "yes", "y"}
        )
        self.text_generated_slot_mode = str(
            os.environ.get("NNUNET_STRUCTCOND_TEXT_GENERATED_SLOT_MODE", "residual")
        ).strip().lower()
        self.text_generated_slot_alpha = float(os.environ.get("NNUNET_STRUCTCOND_TEXT_GENERATED_SLOT_ALPHA", "1.0"))
        self._latest_text_generated_slot_stats: Dict[str, object] = {}
        if self.use_text_generated_slot_head:
            if not self.text_generated_slot_embedding_path:
                raise ValueError(
                    "NNUNET_STRUCTCOND_USE_TEXT_GENERATED_SLOT_HEAD=1 requires NNUNET_STRUCTCOND_TEXT_EMB_PATH"
                )
            if self.text_generated_slot_mode not in {"residual", "replace"}:
                raise ValueError("NNUNET_STRUCTCOND_TEXT_GENERATED_SLOT_MODE must be 'residual' or 'replace'")

        # GradScaler is only useful for float16. BF16 has FP32-like exponent range.
        if self.amp_dtype != torch.float16:
            self.grad_scaler = None

        # Optional epoch snapshot checkpointing:
        # keep normal checkpoint_latest/checkpoint_final and additionally write
        # checkpoint_ep{N}.pth from a chosen starting epoch at fixed intervals.
        self.enable_epoch_checkpoint_snapshots = (
            str(os.environ.get("NNUNET_STRUCTCOND_SAVE_EPOCH_SNAPSHOTS", "0")).lower() in {"1", "true", "yes", "y"}
        )
        self.epoch_checkpoint_snapshot_start = int(
            os.environ.get("NNUNET_STRUCTCOND_SAVE_EPOCH_SNAPSHOTS_FROM_EPOCH", "1000")
        )
        self.epoch_checkpoint_snapshot_every = int(
            os.environ.get("NNUNET_STRUCTCOND_SAVE_EPOCH_SNAPSHOTS_EVERY", "50")
        )
        if self.epoch_checkpoint_snapshot_start < 1:
            raise ValueError("NNUNET_STRUCTCOND_SAVE_EPOCH_SNAPSHOTS_FROM_EPOCH must be >= 1.")
        if self.epoch_checkpoint_snapshot_every < 1:
            raise ValueError("NNUNET_STRUCTCOND_SAVE_EPOCH_SNAPSHOTS_EVERY must be >= 1.")

        # Loss defaults are practical and can be tuned by env vars if needed.
        self.loss_cfg = StructuredLossConfig(
            lambda_ce=float(os.environ.get("NNUNET_STRUCTCOND_LAMBDA_CE", "1.0")),
            lambda_dice=float(os.environ.get("NNUNET_STRUCTCOND_LAMBDA_DICE", "1.0")),
            lambda_cond=float(os.environ.get("NNUNET_STRUCTCOND_LAMBDA_COND", "0.25")),
            lambda_suppress=float(os.environ.get("NNUNET_STRUCTCOND_LAMBDA_SUPPRESS", "0.1")),
            enable_conditional_focus=str(os.environ.get("NNUNET_STRUCTCOND_ENABLE_COND", "1")).lower() in {"1", "true", "yes", "y"},
            enable_suppression=str(os.environ.get("NNUNET_STRUCTCOND_ENABLE_SUPPRESS", "1")).lower() in {"1", "true", "yes", "y"},
            batch_dice=self.configuration_manager.batch_dice,
            smooth=1e-5,
            ddp=self.is_ddp,
        )

        self._ds_loss_weights = np.asarray([1.0], dtype=np.float32)
        self._latest_structured_val_report = {}
        self._group_sample_counter_epoch = np.zeros((self.num_dynamic_groups,), dtype=np.int64)
        self._val_group_cursor = int(self.local_rank) % max(1, self.num_dynamic_groups)
        self._val_step_counter = 0
        self._wandb_module = None
        self._wandb_run = None
        self._wandb_enabled = str(os.environ.get("NNUNET_USE_WANDB", "0")).lower() in {"1", "true", "yes", "y"}

        # Validation speed controls.
        self.val_full_sweep_every = int(os.environ.get("NNUNET_STRUCTCOND_VAL_FULL_EVERY", "1"))
        self.val_full_sweep_batches = int(os.environ.get("NNUNET_STRUCTCOND_VAL_FULL_SWEEP_BATCHES", "0"))
        self.val_full_sweep_epochs = int(os.environ.get("NNUNET_STRUCTCOND_VAL_FULL_SWEEP_EPOCHS", "0"))
        self.val_groups_per_epoch = int(
            os.environ.get("NNUNET_STRUCTCOND_VAL_GROUPS_PER_EPOCH", str(self.num_dynamic_groups))
        )
        self.val_reuse_encoder = str(os.environ.get("NNUNET_STRUCTCOND_VAL_REUSE_ENCODER", "1")).lower() in {"1", "true", "yes", "y"}
        self.val_loss_mode = str(os.environ.get("NNUNET_STRUCTCOND_VAL_LOSS_MODE", "main_only")).strip().lower()
        if self.val_loss_mode not in {"full", "main_only", "none"}:
            raise ValueError("NNUNET_STRUCTCOND_VAL_LOSS_MODE must be one of: full, main_only, none")
        if self.val_groups_per_epoch < 1:
            raise ValueError("NNUNET_STRUCTCOND_VAL_GROUPS_PER_EPOCH must be >= 1")
        if self.val_full_sweep_batches < 0:
            raise ValueError("NNUNET_STRUCTCOND_VAL_FULL_SWEEP_BATCHES must be >= 0")
        if self.val_full_sweep_epochs < 0:
            raise ValueError("NNUNET_STRUCTCOND_VAL_FULL_SWEEP_EPOCHS must be >= 0")

    def _set_batch_size_and_oversample(self):
        override = getattr(self, "_batch_size_override_from_env", 0)
        if override > 0:
            self.configuration_manager.configuration["batch_size"] = int(override)
        super()._set_batch_size_and_oversample()

    def _do_i_compile(self):
        enable = str(os.environ.get("NNUNET_STRUCTCOND_COMPILE", "0")).lower() in {"1", "true", "yes", "y"}
        if not enable:
            return False
        return super()._do_i_compile()

    def _setup_wandb(self) -> None:
        if not self._wandb_enabled or self.local_rank != 0:
            return
        try:
            import wandb  # type: ignore
        except Exception as e:
            self.print_to_log_file(f"[W&B] disabled because import failed: {e}")
            return

        run_id_file = os.path.join(self.output_folder, "wandb_run_id.txt")
        run_id = str(os.environ.get("WANDB_RUN_ID", "")).strip()
        if run_id == "" and os.path.isfile(run_id_file):
            try:
                with open(run_id_file, "r", encoding="utf-8") as f:
                    run_id = f.read().strip()
            except OSError:
                run_id = ""

        project = str(os.environ.get("WANDB_PROJECT", "nnUNet")).strip() or "nnUNet"
        entity = str(os.environ.get("WANDB_ENTITY", "")).strip() or None
        name = str(os.environ.get("WANDB_RUN_NAME", "")).strip()
        if name == "":
            name = (
                f"{self.__class__.__name__}_"
                f"{self.plans_manager.dataset_name}_"
                f"{self.configuration_name}_fold{self.fold}"
            )
        mode = str(os.environ.get("WANDB_MODE", "online")).strip() or "online"
        tags_env = str(os.environ.get("WANDB_TAGS", "")).strip()
        tags = [i.strip() for i in tags_env.split(",") if i.strip()] if tags_env else None

        run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            tags=tags,
            mode=mode,
            dir=self.output_folder,
            id=run_id if run_id != "" else None,
            resume="allow",
            config={
                "trainer": self.__class__.__name__,
                "dataset_name": self.plans_manager.dataset_name,
                "configuration": self.configuration_name,
                "fold": self.fold,
                "batch_size": self.batch_size,
                "num_iterations_per_epoch": self.num_iterations_per_epoch,
                "num_val_iterations_per_epoch": self.num_val_iterations_per_epoch,
                "num_epochs": self.num_epochs,
                "initial_lr": self.initial_lr,
                "weight_decay": self.weight_decay,
                "p_present_group": self.p_present_group,
                "grouping_config_path": self.grouping_config_path if self.grouping_config_path else None,
                "val_full_every": self.val_full_sweep_every,
                "val_full_sweep_batches": self.val_full_sweep_batches,
                "val_full_sweep_epochs": self.val_full_sweep_epochs,
                "val_groups_per_epoch": self.val_groups_per_epoch,
                "val_reuse_encoder": self.val_reuse_encoder,
                "val_loss_mode": self.val_loss_mode,
                "use_text_generated_slot_head": self.use_text_generated_slot_head,
                "text_generated_slot_mode": self.text_generated_slot_mode,
                "text_generated_slot_alpha": self.text_generated_slot_alpha,
                "text_generated_slot_hidden_dim": self.text_generated_slot_hidden_dim,
                "text_generated_slot_use_bias": self.text_generated_slot_use_bias,
                "text_generated_slot_normalize_weight": self.text_generated_slot_normalize_weight,
            },
        )
        self._wandb_module = wandb
        self._wandb_run = run
        try:
            with open(run_id_file, "w", encoding="utf-8") as f:
                f.write(str(run.id))
        except OSError:
            pass
        run_url = getattr(run, "url", None)
        if run_url:
            self.print_to_log_file(f"[W&B] enabled. run_id={run.id}, url={run_url}")
        else:
            self.print_to_log_file(f"[W&B] enabled. run_id={run.id}")

    def _log_wandb_epoch(self) -> None:
        if self._wandb_run is None or self.local_rank != 0:
            return
        logs = self.logger.my_fantastic_logging
        if len(logs.get("train_losses", [])) == 0 or len(logs.get("val_losses", [])) == 0:
            return
        epoch_idx = int(self.current_epoch) - 1
        payload = {
            "epoch": epoch_idx,
            "train/loss": float(logs["train_losses"][-1]),
            "val/loss": float(logs["val_losses"][-1]),
            "val/mean_original31_dice": float(logs["mean_fg_dice"][-1]),
            "lr": float(logs["lrs"][-1]),
            "val/summary_mean_original31_dice": float(
                self._latest_structured_val_report.get("summary", {}).get("mean_original31_dice", logs["mean_fg_dice"][-1])
            ),
        }
        if (
            len(logs.get("epoch_start_timestamps", [])) > 0
            and len(logs.get("epoch_end_timestamps", [])) > 0
        ):
            payload["time/epoch_sec"] = float(
                logs["epoch_end_timestamps"][-1] - logs["epoch_start_timestamps"][-1]
            )
        if self.use_text_generated_slot_head and self._latest_text_generated_slot_stats:
            for key in ("active_slots", "inactive_slots", "mean", "std", "min", "max"):
                value = self._latest_text_generated_slot_stats.get(key, None)
                if value is not None:
                    payload[f"text_generated_slot/{key}"] = float(value)
        self._wandb_run.log(payload, step=epoch_idx)

    def _finish_wandb(self) -> None:
        if self._wandb_run is not None and self.local_rank == 0:
            self._wandb_run.finish()
            self._wandb_run = None

    def initialize(self):
        """Custom initialize to enforce fixed output channels."""
        if self.was_initialized:
            raise RuntimeError("initialize() called more than once.")

        self.num_input_channels = determine_num_input_channels(
            self.plans_manager,
            self.configuration_manager,
            self.dataset_json,
        )

        self.network = self.build_network_architecture(
            self.configuration_manager.network_arch_class_name,
            self.configuration_manager.network_arch_init_kwargs,
            self.configuration_manager.network_arch_init_kwargs_req_import,
            self.num_input_channels,
            self.fixed_num_output_channels,
            self.enable_deep_supervision,
        ).to(self.device)

        self.optimizer, self.lr_scheduler = self.configure_optimizers()

        if self.is_ddp:
            self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
            self.network = DDP(self.network, device_ids=[self.local_rank])

        self.loss = self._build_loss()
        self._ds_loss_weights = self._compute_deep_supervision_weights()
        self.was_initialized = True

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> torch.nn.Module:
        # Enforce fixed output channels regardless of dataset labels.
        del num_output_channels
        backbone = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            NUM_OUTPUT_CHANNELS,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        condition_mode = os.environ.get("NNUNET_STRUCTCOND_CONDITION_MODE", "learned").strip().lower()
        use_text_conditioning = (
            str(os.environ.get("NNUNET_STRUCTCOND_USE_TEXT_CONDITIONING", "0")).lower()
            in {"1", "true", "yes", "y"}
        )
        group_text_matrix = None
        text_emb_path = os.environ.get("NNUNET_STRUCTCOND_TEXT_EMB_PATH", "").strip()
        grouping_config_path = os.environ.get("NNUNET_STRUCTCOND_GROUPING_CONFIG_PATH", "").strip()
        text_fusion = os.environ.get("NNUNET_STRUCTCOND_TEXT_FUSION", "concat_mlp").strip().lower()
        freeze_text_embeddings = (
            str(os.environ.get("NNUNET_STRUCTCOND_FREEZE_TEXT_EMBEDDINGS", "1")).lower()
            in {"1", "true", "yes", "y"}
        )
        use_text_generated_slot_head = (
            str(os.environ.get("NNUNET_STRUCTCOND_USE_TEXT_GENERATED_SLOT_HEAD", "0")).lower()
            in {"1", "true", "yes", "y"}
        )
        text_generated_slot_hidden_dim = int(
            os.environ.get("NNUNET_STRUCTCOND_TEXT_GENERATED_SLOT_HIDDEN_DIM", "512")
        )
        text_generated_slot_use_bias = (
            str(os.environ.get("NNUNET_STRUCTCOND_TEXT_GENERATED_SLOT_USE_BIAS", "1")).lower()
            in {"1", "true", "yes", "y"}
        )
        text_generated_slot_normalize_weight = (
            str(os.environ.get("NNUNET_STRUCTCOND_TEXT_GENERATED_SLOT_NORMALIZE_WEIGHT", "1")).lower()
            in {"1", "true", "yes", "y"}
        )
        text_generated_slot_mode = str(
            os.environ.get("NNUNET_STRUCTCOND_TEXT_GENERATED_SLOT_MODE", "residual")
        ).strip().lower()
        text_generated_slot_alpha = float(os.environ.get("NNUNET_STRUCTCOND_TEXT_GENERATED_SLOT_ALPHA", "1.0"))
        text_generated_slot_embeddings = None
        text_generated_slot_group_to_labels = None
        if use_text_generated_slot_head:
            if not text_emb_path:
                raise ValueError(
                    "NNUNET_STRUCTCOND_USE_TEXT_GENERATED_SLOT_HEAD=1 requires NNUNET_STRUCTCOND_TEXT_EMB_PATH"
                )
            if text_generated_slot_mode not in {"residual", "replace"}:
                raise ValueError("NNUNET_STRUCTCOND_TEXT_GENERATED_SLOT_MODE must be 'residual' or 'replace'")
            text_generated_slot_embeddings, text_generated_slot_group_to_labels, _ = (
                _build_group_slot_text_embedding_matrix(text_emb_path, num_slots=3)
            )
            print(
                "[StructuredConditional-Synapse] "
                f"use_text_generated_slot_head=True  text_emb={text_emb_path}  "
                f"mode={text_generated_slot_mode}  alpha={text_generated_slot_alpha}  "
                f"text_dim={tuple(text_generated_slot_embeddings.shape)}  "
                f"group_to_slot_labels={json.dumps({str(k): list(v) for k, v in text_generated_slot_group_to_labels.items()})}",
                flush=True,
            )
        if use_text_conditioning:
            if not text_emb_path:
                raise ValueError(
                    "NNUNET_STRUCTCOND_USE_TEXT_CONDITIONING=1 requires NNUNET_STRUCTCOND_TEXT_EMB_PATH"
                )
            if text_fusion not in {"concat_mlp", "add", "text_only"}:
                raise ValueError("NNUNET_STRUCTCOND_TEXT_FUSION must be one of: concat_mlp, add, text_only")
            if grouping_config_path:
                group_text_matrix = _build_group_text_matrix_from_label_embeddings(text_emb_path)
            else:
                group_keys = [spec.short_name for spec in DYNAMIC_GROUP_SPECS]
                group_display_names = [spec.display_name for spec in DYNAMIC_GROUP_SPECS]
                group_text_matrix = build_group_text_matrix_from_group_embeddings(
                    text_emb_path,
                    group_keys=group_keys,
                    group_display_names=group_display_names,
                )
            condition_mode = "learned_text"
            print(
                "[StructuredConditional-Synapse] "
                f"use_text_conditioning=True  text_emb={text_emb_path}  text_fusion={text_fusion}  "
                f"group_text_shape={tuple(group_text_matrix.shape)}  freeze_text_embeddings={freeze_text_embeddings}  "
                f"grouping_config_path={grouping_config_path if grouping_config_path else None}",
                flush=True,
            )
        elif condition_mode in ("text", "text_init"):
            if not text_emb_path:
                raise ValueError("NNUNET_STRUCTCOND_CONDITION_MODE=text/text_init requires NNUNET_STRUCTCOND_TEXT_EMB_PATH")
            text_key = os.environ.get("NNUNET_STRUCTCOND_TEXT_EMB_KEY", "per_label_mean_centered").strip()
            group_label_ids = [spec.original_labels for spec in DYNAMIC_GROUP_SPECS]
            group_text_matrix = build_group_text_matrix(text_emb_path, group_label_ids, key=text_key)
            print(f"[StructuredConditional-Synapse] condition_mode={condition_mode}  text_emb={text_emb_path}  "
                  f"key={text_key}  matrix={tuple(group_text_matrix.shape)}", flush=True)
        else:
            print(
                "[StructuredConditional-Synapse] "
                "use_text_conditioning=False  text_emb=None  text_fusion=None  group_text_shape=None",
                flush=True,
            )
        return StructuredConditionalUNet(
            backbone=backbone,
            num_groups=NUM_DYNAMIC_GROUPS,
            num_output_channels=NUM_OUTPUT_CHANNELS,
            cond_dim=64,
            condition_mode=condition_mode,
            group_text_matrix=group_text_matrix,
            text_fusion=text_fusion,
            freeze_text_embeddings=freeze_text_embeddings,
            use_text_generated_slot_head=use_text_generated_slot_head,
            text_generated_slot_embeddings=text_generated_slot_embeddings,
            text_generated_slot_group_to_labels=text_generated_slot_group_to_labels,
            text_generated_slot_hidden_dim=text_generated_slot_hidden_dim,
            text_generated_slot_use_bias=text_generated_slot_use_bias,
            text_generated_slot_normalize_weight=text_generated_slot_normalize_weight,
            text_generated_slot_mode=text_generated_slot_mode,
            text_generated_slot_alpha=text_generated_slot_alpha,
            text_generated_slot_start_channel=COND_SLOT_1_CHANNEL,
            text_generated_slot_num_slots=3,
        )

    def _compute_deep_supervision_weights(self) -> np.ndarray:
        if not self.enable_deep_supervision:
            return np.asarray([1.0], dtype=np.float32)

        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))], dtype=np.float32)

        if self.is_ddp:
            weights[-1] = 1e-6
        else:
            weights[-1] = 0.0

        weights = weights / np.clip(weights.sum(), a_min=1e-12, a_max=None)
        return weights.astype(np.float32)

    def _build_loss(self):
        return StructuredConditionalLoss(self.loss_cfg)

    @staticmethod
    def _to_device_target(target, device: torch.device):
        if isinstance(target, list):
            return [i.to(device, non_blocking=True) for i in target]
        return target.to(device, non_blocking=True)

    def _autocast_context(self):
        if self.device.type == "cuda" and self.amp_dtype is not None:
            return autocast(self.device.type, dtype=self.amp_dtype, enabled=True)
        return dummy_context()

    @staticmethod
    def _output_to_float(output):
        if isinstance(output, (tuple, list)):
            return [x.float() for x in output]
        return output.float()

    def _assert_all_finite(self, value, label: str) -> None:
        if self._all_finite(value):
            return
        self.network.zero_grad(set_to_none=True)
        raise FloatingPointError(
            f"Non-finite {label} detected at epoch {self.current_epoch}. "
            "The optimizer step was aborted to prevent checkpoint corruption."
        )

    def _all_finite(self, value) -> bool:
        values = value if isinstance(value, (tuple, list)) else [value]
        local_finite = torch.ones((), dtype=torch.int32, device=self.device)
        for tensor in values:
            local_finite.mul_(torch.isfinite(tensor).all().to(dtype=torch.int32))

        if self.is_ddp:
            dist.all_reduce(local_finite, op=dist.ReduceOp.MIN)
        return bool(local_finite.item())

    def _skip_nonfinite_gradient_update(self, loss: torch.Tensor, grad_norm) -> dict:
        self.network.zero_grad(set_to_none=True)
        if self.grad_scaler is not None:
            self.grad_scaler.update()
        if not self.skip_nonfinite_grad:
            raise FloatingPointError(
                f"Non-finite gradient norm detected at epoch {self.current_epoch}. "
                "The optimizer step was aborted to prevent checkpoint corruption."
            )
        self._nonfinite_grad_skip_count += 1
        if self.local_rank == 0 and (
            self._nonfinite_grad_skip_count <= 5 or self._nonfinite_grad_skip_count % 50 == 0
        ):
            try:
                grad_norm_value = float(grad_norm.detach().cpu().item())
            except Exception:
                grad_norm_value = float("nan")
            self.print_to_log_file(
                "[StructuredConditional] skipped optimizer step with non-finite "
                f"gradient norm={grad_norm_value} at epoch {self.current_epoch}; "
                f"skip_count={self._nonfinite_grad_skip_count}"
            )
        return {"loss": loss.detach().cpu().numpy()}

    def _sample_group_ids_from_target(self, target_high: torch.Tensor) -> torch.Tensor:
        sampled: List[int] = []
        for b in range(int(target_high.shape[0])):
            present = infer_present_groups_from_segmentation(
                target_high[b],
                ignore_label=self.label_manager.ignore_label,
            )
            group_id = sample_group_id_for_case(
                present_group_ids=sorted(present),
                p_present_group=self.p_present_group,
                rng=self.group_sampling_rng,
            )
            sampled.append(int(group_id))
        return torch.as_tensor(sampled, dtype=torch.long, device=self.device)

    def _extract_group_ids_for_batch(self, batch: dict, target_high: torch.Tensor) -> torch.Tensor:
        group_ids = batch.get("group_id", None)
        if group_ids is None:
            return self._sample_group_ids_from_target(target_high)

        if not torch.is_tensor(group_ids):
            group_ids = torch.as_tensor(group_ids, dtype=torch.long)
        group_ids = group_ids.to(self.device, non_blocking=True).reshape(-1).long()

        if group_ids.numel() != int(target_high.shape[0]):
            raise ValueError(
                f"group_id batch mismatch: got {group_ids.numel()}, expected {int(target_high.shape[0])}"
            )
        return group_ids.clamp(min=0, max=self.num_dynamic_groups - 1)

    def _remap_target_for_group(
        self,
        target,
        group_ids: torch.Tensor,
    ):
        if isinstance(target, list):
            remapped_targets = []
            valid_masks = []
            active_slots = None
            for t in target:
                remapped_t, valid_t, active_t = remap_original_to_structured(
                    t,
                    group_ids=group_ids,
                    ignore_label=self.label_manager.ignore_label,
                )
                remapped_targets.append(remapped_t)
                valid_masks.append(valid_t)
                if active_slots is None:
                    active_slots = active_t
            assert active_slots is not None
            return remapped_targets, valid_masks, active_slots

        remapped, valid, active_slots = remap_original_to_structured(
            target,
            group_ids=group_ids,
            ignore_label=self.label_manager.ignore_label,
        )
        return remapped, valid, active_slots

    def _compute_structured_loss(
        self,
        output,
        remapped_target,
        valid_mask,
        active_slots: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(output, (tuple, list)):
            if not isinstance(remapped_target, list) or not isinstance(valid_mask, list):
                raise RuntimeError("Deep supervision output requires list targets and masks.")

            total = None
            n = min(len(output), len(remapped_target), len(valid_mask), len(self._ds_loss_weights))
            for i in range(n):
                weight = float(self._ds_loss_weights[i])
                if weight == 0.0:
                    continue
                weighted_loss = weight * self.loss(output[i], remapped_target[i], valid_mask[i], active_slots)
                total = weighted_loss if total is None else total + weighted_loss
            if total is None:
                raise RuntimeError("All deep-supervision loss weights are zero.")
            return total

        if isinstance(remapped_target, list) or isinstance(valid_mask, list):
            raise RuntimeError("Non-deep-supervision output received list target or mask.")
        return self.loss(output, remapped_target, valid_mask, active_slots)

    def _compute_structured_loss_main_only(
        self,
        output,
        remapped_target,
        valid_mask,
        active_slots: torch.Tensor,
    ) -> torch.Tensor:
        output_main = get_main_output(output)
        target_main = remapped_target[0] if isinstance(remapped_target, list) else remapped_target
        valid_main = valid_mask[0] if isinstance(valid_mask, list) else valid_mask
        return self.loss(output_main, target_main, valid_main, active_slots)

    def _unwrap_network(self):
        mod = self.network.module if self.is_ddp else self.network
        if hasattr(mod, "_orig_mod"):
            mod = mod._orig_mod
        return mod

    def _get_optimizer_parameters(self):
        if not self._conditioning_optimizer_frozen:
            return list(self.network.parameters())

        frozen_ids = {
            id(parameter)
            for name, parameter in self._unwrap_network().named_parameters()
            if name.startswith("condition_encoder.") or name.startswith("input_affine.")
        }
        parameters = [parameter for parameter in self.network.parameters() if id(parameter) not in frozen_ids]
        if not frozen_ids:
            raise RuntimeError("No conditioning parameters were found to exclude from the optimizer.")
        if not parameters:
            raise RuntimeError("Freezing conditioning left the optimizer with no parameters.")
        return parameters

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self._get_optimizer_parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def _draw_val_group_ids(self, k: int) -> List[int]:
        k = int(max(1, min(k, self.num_dynamic_groups)))
        start = int(self._val_group_cursor)
        out = [int((start + j) % self.num_dynamic_groups) for j in range(k)]
        self._val_group_cursor = int((start + k) % self.num_dynamic_groups)
        return out

    def _get_val_group_ids(self) -> List[int]:
        if self.val_full_sweep_epochs > 0 and int(self.current_epoch) >= int(self.val_full_sweep_epochs):
            full_batch_limit = 0
        else:
            full_batch_limit = self.val_full_sweep_batches

        if full_batch_limit > 0 and self._val_step_counter < full_batch_limit:
            return list(range(self.num_dynamic_groups))
        if self.val_full_sweep_every <= 1:
            return list(range(self.num_dynamic_groups))
        if (int(self.current_epoch) % int(self.val_full_sweep_every)) == 0:
            return list(range(self.num_dynamic_groups))
        if self.val_groups_per_epoch >= self.num_dynamic_groups:
            return list(range(self.num_dynamic_groups))
        return self._draw_val_group_ids(self.val_groups_per_epoch)

    def get_dataloaders(self):
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        deep_supervision_scales = self._get_deep_supervision_scales()
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )

        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            # 2D fallback keeps default dataloader behavior; group IDs will be sampled in train_step.
            dl_tr = nnUNetDataLoader2D(
                dataset_tr,
                self.batch_size,
                initial_patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=tr_transforms,
            )
            dl_val = nnUNetDataLoader2D(
                dataset_val,
                self.batch_size,
                self.configuration_manager.patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=val_transforms,
            )
        else:
            dl_tr = StructuredConditionalDataLoader3D(
                dataset_tr,
                self.batch_size,
                initial_patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=tr_transforms,
                p_present_group=self.p_present_group,
                seed=self.group_sampling_seed + int(self.local_rank),
            )
            dl_val = nnUNetDataLoader3D(
                dataset_val,
                self.batch_size,
                self.configuration_manager.patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=val_transforms,
            )

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )

        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def on_train_start(self):
        super().on_train_start()
        self._setup_wandb()
        self.print_to_log_file(
            "[StructuredConditional] "
            f"fixed_output_channels={self.fixed_num_output_channels}, "
            f"num_dynamic_groups={self.num_dynamic_groups}, "
            f"grouping_config_path={self.grouping_config_path if self.grouping_config_path else None}, "
            f"p_present_group={self.p_present_group:.3f}, "
            f"initial_lr={self.initial_lr}, "
            f"weight_decay={self.weight_decay}, "
            f"save_every={self.save_every}, "
            f"save_epoch_snapshots={self.enable_epoch_checkpoint_snapshots}, "
            f"save_epoch_snapshots_from={self.epoch_checkpoint_snapshot_start}, "
            f"save_epoch_snapshots_every={self.epoch_checkpoint_snapshot_every}, "
            f"loss_cfg={self.loss_cfg}, "
            f"val_full_every={self.val_full_sweep_every}, "
            f"val_full_batches={self.val_full_sweep_batches}, "
            f"val_full_epochs={self.val_full_sweep_epochs}, "
            f"val_groups_per_epoch={self.val_groups_per_epoch}, "
            f"val_reuse_encoder={self.val_reuse_encoder}, "
            f"val_loss_mode={self.val_loss_mode}, "
            f"amp_dtype={self.amp_dtype_name}, "
            f"loss_in_fp32={self.loss_in_fp32}, "
            f"grad_clip_norm={self.grad_clip_norm}, "
            f"reset_optimizer_on_load={self.reset_optimizer_on_load}, "
            f"freeze_conditioning_on_load={self.freeze_conditioning_on_load}, "
            f"use_text_generated_slot_head={self.use_text_generated_slot_head}, "
            f"text_generated_slot_mode={self.text_generated_slot_mode}, "
            f"text_generated_slot_alpha={self.text_generated_slot_alpha}, "
            f"text_generated_slot_hidden_dim={self.text_generated_slot_hidden_dim}, "
            f"text_generated_slot_use_bias={self.text_generated_slot_use_bias}, "
            f"text_generated_slot_normalize_weight={self.text_generated_slot_normalize_weight}"
        )

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        checkpoint_to_load = filename_or_checkpoint
        if self.reset_optimizer_on_load or self.freeze_conditioning_on_load:
            if not self.was_initialized:
                self.initialize()
            if isinstance(filename_or_checkpoint, str):
                try:
                    checkpoint_to_load = torch.load(
                        filename_or_checkpoint,
                        map_location=self.device,
                        weights_only=False,
                    )
                except TypeError:
                    checkpoint_to_load = torch.load(filename_or_checkpoint, map_location=self.device)
            else:
                checkpoint_to_load = dict(filename_or_checkpoint)

            # The recovery optimizer may intentionally use a different parameter
            # set, so loading the saved optimizer before rebuilding it can fail.
            checkpoint_to_load = dict(checkpoint_to_load)
            checkpoint_to_load["optimizer_state"] = self.optimizer.state_dict()
            checkpoint_to_load["grad_scaler_state"] = None

        super().load_checkpoint(checkpoint_to_load)
        if self.freeze_conditioning_on_load:
            frozen_names = [
                name
                for name, _ in self._unwrap_network().named_parameters()
                if name.startswith("condition_encoder.") or name.startswith("input_affine.")
            ]
            if not frozen_names:
                raise RuntimeError(
                    "NNUNET_STRUCTCOND_FREEZE_CONDITIONING_ON_LOAD was requested, "
                    "but no conditioning parameters were found."
                )
            self._conditioning_optimizer_frozen = True
            self.print_to_log_file(
                f"[StructuredConditional] Excluding {len(frozen_names)} conditioning parameter tensors "
                "from optimizer updates while retaining DDP gradient participation."
            )
        if self.reset_optimizer_on_load or self.freeze_conditioning_on_load:
            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            self.print_to_log_file(
                "[StructuredConditional] Reset optimizer and LR scheduler after loading checkpoint."
            )

    def _should_write_epoch_snapshot(self, epoch_number: int) -> bool:
        if not self.enable_epoch_checkpoint_snapshots:
            return False
        if epoch_number < self.epoch_checkpoint_snapshot_start:
            return False
        return ((epoch_number - self.epoch_checkpoint_snapshot_start) % self.epoch_checkpoint_snapshot_every) == 0

    def save_checkpoint(self, filename: str) -> None:
        super().save_checkpoint(filename)
        if self.disable_checkpointing:
            return

        is_periodic_latest = filename.endswith("checkpoint_latest.pth")
        is_final = filename.endswith("checkpoint_final.pth")
        if not (is_periodic_latest or is_final):
            return

        epoch_number = int(self.current_epoch) + 1
        if not self._should_write_epoch_snapshot(epoch_number):
            return

        snapshot = os.path.join(self.output_folder, f"checkpoint_ep{epoch_number}.pth")
        if os.path.abspath(snapshot) == os.path.abspath(filename):
            return
        super().save_checkpoint(snapshot)

    def on_validation_epoch_start(self):
        self._val_step_counter = 0
        super().on_validation_epoch_start()

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._to_device_target(batch["target"], self.device)
        target_high = target[0] if isinstance(target, list) else target

        group_ids = self._extract_group_ids_for_batch(batch, target_high)
        bincount = np.bincount(group_ids.detach().cpu().numpy(), minlength=self.num_dynamic_groups)
        self._group_sample_counter_epoch += bincount.astype(np.int64)

        remapped_target, valid_mask, active_slots = self._remap_target_for_group(target, group_ids)

        self.network.zero_grad(set_to_none=True)
        with self._autocast_context():
            output = self.network(data, group_ids)
        text_generated_slot_stats: Dict[str, object] = {}
        if self.use_text_generated_slot_head:
            text_generated_slot_stats = dict(
                getattr(self._unwrap_network(), "latest_text_generated_slot_stats", {})
            )
        self._assert_all_finite(output, "network output")

        loss_output = self._output_to_float(output) if self.loss_in_fp32 else output
        loss = self._compute_structured_loss(loss_output, remapped_target, valid_mask, active_slots)
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

        return {
            "loss": loss.detach().cpu().numpy(),
            "text_generated_slot_active_slots": int(text_generated_slot_stats.get("active_slots", 0)),
            "text_generated_slot_inactive_slots": int(text_generated_slot_stats.get("inactive_slots", 0)),
            "text_generated_slot_logit_mean": float(text_generated_slot_stats.get("mean", 0.0)),
            "text_generated_slot_logit_std": float(text_generated_slot_stats.get("std", 0.0)),
            "text_generated_slot_logit_min": float(text_generated_slot_stats.get("min", 0.0)),
            "text_generated_slot_logit_max": float(text_generated_slot_stats.get("max", 0.0)),
        }

    def on_train_epoch_end(self, train_outputs: List[dict]):
        super().on_train_epoch_end(train_outputs)
        if self.use_text_generated_slot_head and train_outputs:
            slot_stats = {
                "active_slots": float(
                    np.mean([float(o.get("text_generated_slot_active_slots", 0.0)) for o in train_outputs])
                ),
                "inactive_slots": float(
                    np.mean([float(o.get("text_generated_slot_inactive_slots", 0.0)) for o in train_outputs])
                ),
                "mean": float(
                    np.mean([float(o.get("text_generated_slot_logit_mean", 0.0)) for o in train_outputs])
                ),
                "std": float(
                    np.mean([float(o.get("text_generated_slot_logit_std", 0.0)) for o in train_outputs])
                ),
                "min": float(
                    np.min([float(o.get("text_generated_slot_logit_min", 0.0)) for o in train_outputs])
                ),
                "max": float(
                    np.max([float(o.get("text_generated_slot_logit_max", 0.0)) for o in train_outputs])
                ),
            }
            self._latest_text_generated_slot_stats = slot_stats
            self.print_to_log_file("[TextGeneratedSlot] epoch summary " + json.dumps(slot_stats))
        self.print_to_log_file(
            "[StructuredConditional] train group sample counts this epoch: "
            + json.dumps(self._group_sample_counter_epoch.tolist())
        )
        self._group_sample_counter_epoch[:] = 0

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._to_device_target(batch["target"], self.device)

        accum = empty_validation_accumulators()
        losses: List[float] = []

        self._val_step_counter += 1
        batch_size = int(data.shape[0])
        group_ids_list = self._get_val_group_ids()

        reuse_encoder = self.val_reuse_encoder
        skips = None
        if reuse_encoder:
            mod = self._unwrap_network()
            if hasattr(mod, "encode"):
                with self._autocast_context():
                    skips = mod.encode(data)
            else:
                reuse_encoder = False

        for group_id in group_ids_list:
            group_ids = torch.full((batch_size,), int(group_id), dtype=torch.long, device=self.device)
            remapped_target, valid_mask, active_slots = self._remap_target_for_group(target, group_ids)

            with self._autocast_context():
                if reuse_encoder and skips is not None:
                    output = self._unwrap_network().decode_from_skips(skips, group_ids)
                else:
                    output = self.network(data, group_ids)
            self._assert_all_finite(output, "validation network output")

            loss_output = self._output_to_float(output) if self.loss_in_fp32 else output
            if self.val_loss_mode == "full":
                loss = self._compute_structured_loss(loss_output, remapped_target, valid_mask, active_slots)
            elif self.val_loss_mode == "main_only":
                loss = self._compute_structured_loss_main_only(
                    loss_output, remapped_target, valid_mask, active_slots
                )
            else:
                loss = get_main_output(loss_output).reshape(-1)[0] * 0.0
            self._assert_all_finite(loss, "validation loss")

            losses.append(float(loss.detach().cpu().item()))

            output_main = get_main_output(output)
            target_main = remapped_target[0] if isinstance(remapped_target, list) else remapped_target
            valid_main = valid_mask[0] if isinstance(valid_mask, list) else valid_mask

            (
                class_tp,
                class_fp,
                class_fn,
                cond_tp,
                cond_fp,
                cond_fn,
                merged_tp,
                merged_fp,
                merged_fn,
            ) = compute_group_confusion_from_logits(
                output_main,
                target_main,
                valid_main,
                group_id=group_id,
            )

            accum["class_tp"] += class_tp
            accum["class_fp"] += class_fp
            accum["class_fn"] += class_fn
            accum["cond_tp"][group_id] += cond_tp
            accum["cond_fp"][group_id] += cond_fp
            accum["cond_fn"][group_id] += cond_fn
            accum["merged_cond_tp"][group_id] += merged_tp[0]
            accum["merged_cond_fp"][group_id] += merged_fp[0]
            accum["merged_cond_fn"][group_id] += merged_fn[0]

        mean_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

        return {
            "loss": np.float32(mean_loss),
            "class_tp": accum["class_tp"],
            "class_fp": accum["class_fp"],
            "class_fn": accum["class_fn"],
            "cond_tp": accum["cond_tp"],
            "cond_fp": accum["cond_fp"],
            "cond_fn": accum["cond_fn"],
            "merged_cond_tp": accum["merged_cond_tp"],
            "merged_cond_fp": accum["merged_cond_fp"],
            "merged_cond_fn": accum["merged_cond_fn"],
        }

    @staticmethod
    def _ddp_sum_array(array_value: np.ndarray) -> np.ndarray:
        world_size = dist.get_world_size()
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, array_value)
        return np.stack(gathered, axis=0).sum(axis=0)

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs = collate_outputs(val_outputs)

        class_tp = np.sum(outputs["class_tp"], axis=0)
        class_fp = np.sum(outputs["class_fp"], axis=0)
        class_fn = np.sum(outputs["class_fn"], axis=0)
        cond_tp = np.sum(outputs["cond_tp"], axis=0)
        cond_fp = np.sum(outputs["cond_fp"], axis=0)
        cond_fn = np.sum(outputs["cond_fn"], axis=0)
        merged_cond_tp = np.sum(outputs["merged_cond_tp"], axis=0)
        merged_cond_fp = np.sum(outputs["merged_cond_fp"], axis=0)
        merged_cond_fn = np.sum(outputs["merged_cond_fn"], axis=0)

        if self.is_ddp:
            class_tp = self._ddp_sum_array(class_tp)
            class_fp = self._ddp_sum_array(class_fp)
            class_fn = self._ddp_sum_array(class_fn)
            cond_tp = self._ddp_sum_array(cond_tp)
            cond_fp = self._ddp_sum_array(cond_fp)
            cond_fn = self._ddp_sum_array(cond_fn)
            merged_cond_tp = self._ddp_sum_array(merged_cond_tp)
            merged_cond_fp = self._ddp_sum_array(merged_cond_fp)
            merged_cond_fn = self._ddp_sum_array(merged_cond_fn)

            losses_val = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_val, outputs["loss"])
            loss_here = float(np.vstack(losses_val).mean())
        else:
            loss_here = float(np.mean(outputs["loss"]))

        report = build_validation_report(
            class_tp=class_tp,
            class_fp=class_fp,
            class_fn=class_fn,
            cond_tp=cond_tp,
            cond_fp=cond_fp,
            cond_fn=cond_fn,
            merged_cond_tp=merged_cond_tp,
            merged_cond_fp=merged_cond_fp,
            merged_cond_fn=merged_cond_fn,
        )
        self._latest_structured_val_report = report

        mean_active_dice = float(report["summary"]["mean_original31_dice"])
        dice_foreground = report["original31_dice"]

        self.logger.log("mean_fg_dice", mean_active_dice, self.current_epoch)
        self.logger.log("dice_per_class_or_region", dice_foreground, self.current_epoch)
        self.logger.log("val_losses", loss_here, self.current_epoch)

        self.print_to_log_file("[StructuredConditional][val] " + json.dumps(report["summary"], sort_keys=True))

    def on_epoch_end(self):
        super().on_epoch_end()
        self._log_wandb_epoch()

    def on_train_end(self):
        try:
            super().on_train_end()
        finally:
            self._finish_wandb()

    def perform_actual_validation(self, save_probabilities: bool = False):
        # The structured 9-channel conditional head is incompatible with nnU-Net's
        # default flat sliding-window validator, which pre-allocates one channel per
        # dataset class (14 for Synapse) and collides with the 9 structured channels.
        # Best-checkpoint selection already happens via the per-epoch structured
        # validation; final original-label scoring is done with the structured
        # multi-group inference + reconstruction (inference_structured_conditional_synapse).
        self.print_to_log_file(
            "[AnchorSlot-Synapse] Skipping nnU-Net flat perform_actual_validation "
            "(incompatible with the structured conditional head). Run the structured "
            "inference/reconstruction script for final validation scoring."
        )
        return

    @torch.no_grad()
    def infer_logits_for_group(self, image: torch.Tensor, group_id: int, use_amp: bool = True) -> torch.Tensor:
        self.network.eval()
        return predict_logits_for_group(self.network, image, group_id=group_id, use_amp=use_amp)

    @torch.no_grad()
    def infer_logits_all_groups(self, image: torch.Tensor, use_amp: bool = True):
        self.network.eval()
        return predict_logits_all_groups(self.network, image, use_amp=use_amp)

    @torch.no_grad()
    def infer_reconstruct_original_all_groups(self, image: torch.Tensor, use_amp: bool = True):
        self.network.eval()
        logits_by_group = predict_logits_all_groups(self.network, image, use_amp=use_amp)
        return reconstruct_original_labels_from_all_groups(logits_by_group)
