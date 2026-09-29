from __future__ import annotations

import json
import os

import torch

from nnunetv2.training.loss.dynamic_distribution_weighting import (
    DynamicDistributionWeighting,
)

from .label_mapping_er_dynamic import (
    COND_SLOT_1_CHANNEL,
    COND_SLOT_3_CHANNEL,
    FIXED_OUTPUT_TO_ORIGINAL,
    NUM_OUTPUT_CHANNELS,
    OTHER_CHANNEL,
    get_group_spec,
)
from .structured_loss_er_dynamic_ddw import AnchorSlotDynamicDistributionLoss
from .trainer_structured_conditional_er_dynamic import (
    get_main_output,
    nnUNetTrainerStructuredConditionalERDynamic,
)


class AnchorSlotDDWMixin:
    """Backbone-independent semantic, group-aware distribution weighting."""

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
        self.ddw_ema = float(os.environ.get("NNUNET_STRUCTCOND_DDW_EMA", "0.99"))
        self.ddw_alpha = float(os.environ.get("NNUNET_STRUCTCOND_DDW_ALPHA", "0.25"))
        self.ddw_min_weight = float(
            os.environ.get("NNUNET_STRUCTCOND_DDW_MIN_WEIGHT", "0.5")
        )
        self.ddw_max_weight = float(
            os.environ.get("NNUNET_STRUCTCOND_DDW_MAX_WEIGHT", "2.0")
        )
        self.ddw_warmup_epochs = int(
            os.environ.get("NNUNET_STRUCTCOND_DDW_WARMUP_EPOCHS", "10")
        )
        self.ddw_ramp_epochs = int(
            os.environ.get("NNUNET_STRUCTCOND_DDW_RAMP_EPOCHS", "20")
        )
        self.ddw_eps = float(os.environ.get("NNUNET_STRUCTCOND_DDW_EPS", "1e-6"))
        if self.loss_cfg.class_balance_mode not in {"", "none", "off", "0", "false"}:
            raise ValueError(
                "DDW must not be combined with static class balancing; "
                "NNUNET_STRUCTCOND_CLASS_BALANCE_MODE must be 'none'."
            )
        self.ddw_tracker: DynamicDistributionWeighting | None = None
        self._ddw_current_group_ids: torch.Tensor | None = None

    def _build_loss(self):
        self.ddw_tracker = DynamicDistributionWeighting(
            32,
            ema=self.ddw_ema,
            alpha=self.ddw_alpha,
            min_weight=self.ddw_min_weight,
            max_weight=self.ddw_max_weight,
            eps=self.ddw_eps,
            warmup_epochs=self.ddw_warmup_epochs,
            ramp_epochs=self.ddw_ramp_epochs,
            neutral_class_ids=(0,),
        ).to(self.device)
        return AnchorSlotDynamicDistributionLoss(self.loss_cfg)

    def _extract_group_ids_for_batch(
        self, batch: dict, target_high: torch.Tensor
    ) -> torch.Tensor:
        group_ids = super()._extract_group_ids_for_batch(batch, target_high)
        self._ddw_current_group_ids = group_ids
        return group_ids

    @torch.no_grad()
    def _update_ddw_from_main_output(
        self,
        output,
        valid_mask,
        active_slots: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> None:
        if self.ddw_tracker is None:
            raise RuntimeError("DDW tracker was not initialized")
        logits = get_main_output(output).detach().float()
        valid = valid_mask[0] if isinstance(valid_mask, list) else valid_mask
        valid = valid[:, 0].bool()

        class_mask = torch.ones(
            (logits.shape[0], NUM_OUTPUT_CHANNELS), dtype=torch.bool, device=logits.device
        )
        class_mask[:, COND_SLOT_1_CHANNEL] = active_slots[:, 0]
        class_mask[:, COND_SLOT_1_CHANNEL + 1] = active_slots[:, 1]
        class_mask[:, COND_SLOT_3_CHANNEL] = active_slots[:, 2]
        expanded = class_mask.view(class_mask.shape[0], class_mask.shape[1], *([1] * (logits.ndim - 2)))
        prediction = logits.masked_fill(~expanded, -1e4).argmax(dim=1)

        counts = torch.zeros(32, dtype=torch.float32, device=logits.device)
        denominators = torch.zeros_like(counts)
        for sample_index in range(logits.shape[0]):
            sample_valid = valid[sample_index]
            num_valid = float(sample_valid.sum().item())
            if num_valid <= 0:
                continue
            spec = get_group_spec(int(group_ids[sample_index].item()))
            output_to_original = dict(FIXED_OUTPUT_TO_ORIGINAL)
            output_to_original.update(
                {
                    COND_SLOT_1_CHANNEL + slot: int(original_label)
                    for slot, original_label in enumerate(spec.original_labels)
                }
            )
            predicted = prediction[sample_index]
            for output_channel, original_label in output_to_original.items():
                counts[original_label] += ((predicted == output_channel) & sample_valid).sum()
                denominators[original_label] += num_valid
        self.ddw_tracker.update(counts, denominators)

    def _sample_output_weights(
        self, group_ids: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        if self.ddw_tracker is None:
            raise RuntimeError("DDW tracker was not initialized")
        semantic_weights = self.ddw_tracker.weights(self.current_epoch, device=device)
        result = torch.ones(
            (group_ids.numel(), NUM_OUTPUT_CHANNELS), dtype=torch.float32, device=device
        )
        for output_channel, original_label in FIXED_OUTPUT_TO_ORIGINAL.items():
            result[:, output_channel] = semantic_weights[original_label]
        for sample_index, group_id in enumerate(group_ids.tolist()):
            spec = get_group_spec(int(group_id))
            for slot, original_label in enumerate(spec.original_labels):
                result[sample_index, COND_SLOT_1_CHANNEL + slot] = semantic_weights[
                    original_label
                ]
        result[:, OTHER_CHANNEL] = 1.0
        return result

    def _compute_structured_loss(
        self, output, remapped_target, valid_mask, active_slots: torch.Tensor
    ) -> torch.Tensor:
        group_ids = self._ddw_current_group_ids
        if self.network.training:
            if group_ids is None or group_ids.numel() != get_main_output(output).shape[0]:
                raise RuntimeError("AnchorSlot DDW did not receive the current training group IDs")
            self._update_ddw_from_main_output(output, valid_mask, active_slots, group_ids)
            self.loss.set_sample_class_weights(
                self._sample_output_weights(group_ids, get_main_output(output).device)
            )
        else:
            self.loss.set_sample_class_weights(None)
        return super()._compute_structured_loss(
            output, remapped_target, valid_mask, active_slots
        )

    def _compute_structured_loss_main_only(
        self, output, remapped_target, valid_mask, active_slots: torch.Tensor
    ) -> torch.Tensor:
        # Validation metrics remain directly comparable to the control: DDW is
        # a training objective, not a redefinition of validation loss/metrics.
        self.loss.set_sample_class_weights(None)
        return super()._compute_structured_loss_main_only(
            output, remapped_target, valid_mask, active_slots
        )

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        if self.ddw_tracker is not None:
            weights = self.ddw_tracker.weights(self.current_epoch).detach().cpu()
            self.print_to_log_file(
                "[AnchorSlotDDW] epoch summary "
                + json.dumps(
                    {
                        "updates": int(self.ddw_tracker.num_updates.item()),
                        "weight_min": float(weights[1:].min().item()),
                        "weight_max": float(weights[1:].max().item()),
                        "weight_mean": float(weights[1:].mean().item()),
                    }
                )
            )

    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file(
            "[AnchorSlotDDW] "
            f"ema={self.ddw_ema}, alpha={self.ddw_alpha}, "
            f"bounds=({self.ddw_min_weight}, {self.ddw_max_weight}), "
            f"warmup_epochs={self.ddw_warmup_epochs}, ramp_epochs={self.ddw_ramp_epochs}, "
            "weighted_term=CE_only, semantic_space=original_31_classes, "
            "background_weight=1, other_weight=1"
        )

    def _ddw_config(self):
        return {key: getattr(self, "ddw_" + key) for key in (
            "ema", "alpha", "min_weight", "max_weight", "warmup_epochs", "ramp_epochs", "eps"
        )}

    def save_checkpoint(self, filename: str) -> None:
        # Tracker state is not a network parameter. Keep it with the epoch and
        # optimizer in every save, including snapshots, using atomic replacement.
        if self.local_rank != 0 or self.disable_checkpointing:
            return
        if self.ddw_tracker is None:
            raise RuntimeError("Cannot save an uninitialized DDW tracker")
        checkpoint = {
            "network_weights": self._unwrap_network().state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "grad_scaler_state": self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
            "logging": self.logger.get_checkpoint(),
            "_best_ema": self._best_ema,
            "current_epoch": self.current_epoch + 1,
            "init_args": self.my_init_kwargs,
            "trainer_name": self.__class__.__name__,
            "inference_allowed_mirroring_axes": self.inference_allowed_mirroring_axes,
            "ddw_state": {"version": 1, "config": self._ddw_config(),
                          "tracker": self.ddw_tracker.state_dict()},
        }
        paths = [os.fspath(filename)]
        if (paths[0].endswith(("checkpoint_latest.pth", "checkpoint_final.pth"))
                and self._should_write_epoch_snapshot(self.current_epoch + 1)):
            snapshot = os.path.join(self.output_folder, f"checkpoint_ep{self.current_epoch + 1}.pth")
            if os.path.abspath(snapshot) != os.path.abspath(paths[0]):
                paths.append(snapshot)
        for path in paths:
            temporary = f"{path}.tmp-{os.getpid()}"
            try:
                torch.save(checkpoint, temporary)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        if isinstance(filename_or_checkpoint, (str, os.PathLike)):
            checkpoint = torch.load(filename_or_checkpoint, map_location="cpu", weights_only=False)
        else:
            checkpoint = filename_or_checkpoint
        state = checkpoint.get("ddw_state")
        if state is None:
            raise ValueError("DDW resume requires saved ddw_state. For a new fine-tune, load full network weights only.")
        if state.get("version") != 1 or state.get("config") != self._ddw_config():
            raise ValueError("DDW checkpoint configuration differs from the requested resume configuration")
        super().load_checkpoint(checkpoint)
        self.ddw_tracker.load_state_dict(state["tracker"], strict=True)
        self.loss.set_sample_class_weights(None)
        self._ddw_current_group_ids = None


class nnUNetTrainerStructuredConditionalERDynamicDDW(
    AnchorSlotDDWMixin, nnUNetTrainerStructuredConditionalERDynamic
):
    """PlainConvUNet AnchorSlot with DDW on the masked CE term only."""
