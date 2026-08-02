from __future__ import annotations

import os
from typing import List, Tuple, Union

import torch
from nnunet_mednext import create_mednext_v1
from torch import nn
from torch._dynamo import OptimizedModule

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


class nnUNetTrainerMedNeXt32OutNoCond(nnUNetTrainer):
    """
    MedNeXt backbone trainer for the original 32-channel CellMap label space.

    This trainer intentionally has no structured-conditional group input and
    uses nnU-Net's standard multiclass loss/dataloaders.
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

        self.mednext_model_id = str(
            os.environ.get(
                "NNUNET_MEDNEXT_MODEL_ID",
                os.environ.get("NNUNET_STRUCTCOND_MEDNEXT_MODEL_ID", "S"),
            )
        ).strip().upper()
        self.mednext_kernel_size = int(
            os.environ.get(
                "NNUNET_MEDNEXT_KERNEL_SIZE",
                os.environ.get("NNUNET_STRUCTCOND_MEDNEXT_KERNEL_SIZE", "3"),
            )
        )
        self.fixed_num_output_channels = int(os.environ.get("NNUNET_MEDNEXT_NUM_OUTPUT_CHANNELS", "32"))

        if self.mednext_model_id not in {"S", "B", "M", "L"}:
            raise ValueError("NNUNET_MEDNEXT_MODEL_ID must be one of: S, B, M, L")
        if self.mednext_kernel_size not in {3, 5, 7}:
            raise ValueError("NNUNET_MEDNEXT_KERNEL_SIZE must be one of: 3, 5, 7")
        if self.fixed_num_output_channels < 2:
            raise ValueError("NNUNET_MEDNEXT_NUM_OUTPUT_CHANNELS must be >= 2")

        self.initial_lr = float(
            os.environ.get("NNUNET_MEDNEXT_INITIAL_LR", os.environ.get("NNUNET_STRUCTCOND_INITIAL_LR", "0.001"))
        )
        self.weight_decay = float(
            os.environ.get("NNUNET_MEDNEXT_WEIGHT_DECAY", os.environ.get("NNUNET_STRUCTCOND_WEIGHT_DECAY", str(self.weight_decay)))
        )
        self.num_epochs = int(
            os.environ.get("NNUNET_MEDNEXT_NUM_EPOCHS", os.environ.get("NNUNET_STRUCTCOND_NUM_EPOCHS", str(self.num_epochs)))
        )
        if self.initial_lr <= 0:
            raise ValueError("NNUNET_MEDNEXT_INITIAL_LR must be > 0")
        if self.weight_decay < 0:
            raise ValueError("NNUNET_MEDNEXT_WEIGHT_DECAY must be >= 0")
        if self.num_epochs < 1:
            raise ValueError("NNUNET_MEDNEXT_NUM_EPOCHS must be >= 1")

        # Optional epoch snapshot checkpointing:
        # keep normal checkpoint_latest/checkpoint_final and additionally write
        # checkpoint_ep{N}.pth from a chosen starting epoch at fixed intervals.
        self.enable_epoch_checkpoint_snapshots = (
            str(
                os.environ.get(
                    "NNUNET_MEDNEXT_SAVE_EPOCH_SNAPSHOTS",
                    os.environ.get("NNUNET_STRUCTCOND_SAVE_EPOCH_SNAPSHOTS", "0"),
                )
            ).lower() in {"1", "true", "yes", "y"}
        )
        self.epoch_checkpoint_snapshot_start = int(
            os.environ.get(
                "NNUNET_MEDNEXT_SAVE_EPOCH_SNAPSHOTS_FROM_EPOCH",
                os.environ.get("NNUNET_STRUCTCOND_SAVE_EPOCH_SNAPSHOTS_FROM_EPOCH", "1000"),
            )
        )
        self.epoch_checkpoint_snapshot_every = int(
            os.environ.get(
                "NNUNET_MEDNEXT_SAVE_EPOCH_SNAPSHOTS_EVERY",
                os.environ.get("NNUNET_STRUCTCOND_SAVE_EPOCH_SNAPSHOTS_EVERY", "50"),
            )
        )
        if self.epoch_checkpoint_snapshot_start < 1:
            raise ValueError("NNUNET_MEDNEXT_SAVE_EPOCH_SNAPSHOTS_FROM_EPOCH must be >= 1")
        if self.epoch_checkpoint_snapshot_every < 1:
            raise ValueError("NNUNET_MEDNEXT_SAVE_EPOCH_SNAPSHOTS_EVERY must be >= 1")

        self._wandb_module = None
        self._wandb_run = None
        self._wandb_enabled = str(os.environ.get("NNUNET_USE_WANDB", "0")).lower() in {"1", "true", "yes", "y"}

    def _set_batch_size_and_oversample(self):
        batch_size_override = int(
            os.environ.get("NNUNET_MEDNEXT_BATCH_SIZE", os.environ.get("NNUNET_STRUCTCOND_BATCH_SIZE", "0"))
        )
        if batch_size_override < 0:
            raise ValueError("NNUNET_MEDNEXT_BATCH_SIZE must be >= 0")
        if batch_size_override > 0:
            self.configuration_manager.configuration["batch_size"] = int(batch_size_override)
        super()._set_batch_size_and_oversample()

    def _do_i_compile(self):
        enable = str(
            os.environ.get("NNUNET_MEDNEXT_COMPILE", os.environ.get("NNUNET_STRUCTCOND_COMPILE", "0"))
        ).lower() in {"1", "true", "yes", "y"}
        if not enable:
            return False
        return super()._do_i_compile()

    def initialize(self):
        """Custom initialize to enforce the requested fixed output channels."""
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
            self.network = torch.nn.parallel.DistributedDataParallel(self.network, device_ids=[self.local_rank])

        self.loss = self._build_loss()
        self.was_initialized = True

    def on_train_start(self):
        super().on_train_start()
        self._setup_wandb()
        expected_heads = int(self.label_manager.num_segmentation_heads)
        if expected_heads != int(self.fixed_num_output_channels):
            self.print_to_log_file(
                "[MedNeXt32OutNoCond][warning] "
                f"label_manager.num_segmentation_heads={expected_heads}, "
                f"fixed_num_output_channels={self.fixed_num_output_channels}"
            )
        self.print_to_log_file(
            "[MedNeXt32OutNoCond] "
            f"mednext_model_id={self.mednext_model_id}, "
            f"mednext_kernel_size={self.mednext_kernel_size}, "
            f"num_output_channels={self.fixed_num_output_channels}, "
            f"initial_lr={self.initial_lr}, "
            f"weight_decay={self.weight_decay}, "
            f"num_epochs={self.num_epochs}, "
            f"batch_size={self.configuration_manager.batch_size}, "
            f"save_every={self.save_every}, "
            f"save_epoch_snapshots={self.enable_epoch_checkpoint_snapshots}, "
            f"save_epoch_snapshots_from={self.epoch_checkpoint_snapshot_start}, "
            f"save_epoch_snapshots_every={self.epoch_checkpoint_snapshot_every}"
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
                "mednext_model_id": self.mednext_model_id,
                "mednext_kernel_size": self.mednext_kernel_size,
                "num_output_channels": self.fixed_num_output_channels,
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
            "val/mean_fg_dice": float(logs["mean_fg_dice"][-1]),
            "lr": float(logs["lrs"][-1]),
        }

        dice_values = logs.get("dice_per_class_or_region", [])
        if len(dice_values) > 0:
            for idx, dice_value in enumerate(dice_values[-1], start=1):
                payload[f"val/dice_per_class/{idx:02d}"] = float(dice_value)

        if len(logs.get("epoch_start_timestamps", [])) > 0 and len(logs.get("epoch_end_timestamps", [])) > 0:
            payload["time/epoch_sec"] = float(
                logs["epoch_end_timestamps"][-1] - logs["epoch_start_timestamps"][-1]
            )
        self._wandb_run.log(payload, step=epoch_idx)

    def _finish_wandb(self) -> None:
        if self._wandb_run is not None and self.local_rank == 0:
            self._wandb_run.finish()
            self._wandb_run = None

    def on_epoch_end(self):
        super().on_epoch_end()
        self._log_wandb_epoch()

    def on_train_end(self):
        try:
            super().on_train_end()
        finally:
            self._finish_wandb()

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        del architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import
        model_id = str(
            os.environ.get(
                "NNUNET_MEDNEXT_MODEL_ID",
                os.environ.get("NNUNET_STRUCTCOND_MEDNEXT_MODEL_ID", "S"),
            )
        ).strip().upper()
        kernel_size = int(
            os.environ.get(
                "NNUNET_MEDNEXT_KERNEL_SIZE",
                os.environ.get("NNUNET_STRUCTCOND_MEDNEXT_KERNEL_SIZE", "3"),
            )
        )
        out_channels = int(os.environ.get("NNUNET_MEDNEXT_NUM_OUTPUT_CHANNELS", str(num_output_channels)))
        network = create_mednext_v1(
            num_input_channels=int(num_input_channels),
            num_classes=int(out_channels),
            model_id=model_id,
            kernel_size=int(kernel_size),
            # Keep DS heads instantiated so checkpoints trained with DS heads can be loaded in inference.
            deep_supervision=True,
        )
        # Toggle DS outputs according to caller intent after module construction.
        if hasattr(network, "do_ds"):
            network.do_ds = bool(enable_deep_supervision)
        if hasattr(network, "outside_block_checkpointing"):
            network.outside_block_checkpointing = True
        return network

    def set_deep_supervision_enabled(self, enabled: bool):
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        if hasattr(mod, "do_ds"):
            mod.do_ds = bool(enabled)
            return
        super().set_deep_supervision_enabled(enabled)
