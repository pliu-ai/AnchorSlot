"""Opt-in fold-0 screen: single shared encoder, decoder, and nine-channel head."""
from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .label_mapping_synapse import DYNAMIC_GROUP_SPECS
from .network_structured_conditional import get_main_output
from .synapse_reconstruction import masked_probabilities, reconstruct_scores, original_label_loss
from .trainer_synapse_consistency import nnUNetTrainerSynapseMasked


ARMS = ("control", "no_suppression", "lr003", "all_query_control", "reconstruction")


def screen_step_output(loss, structured_loss=None, **diagnostics):
    """Stable reporting schema for both frozen and newer parent trainers."""
    result = {
        "loss": loss,
        "structured_loss": loss if structured_loss is None else structured_loss,
        "text_contrast_loss": np.zeros_like(loss),
        "text_contrast_num_samples": 0,
        "text_contrast_num_labels": 0,
        "text_contrast_sampled_label_ids": [],
        "text_contrast_missing_label_ids": [],
        "text_generated_slot_active_slots": 0,
        "text_generated_slot_inactive_slots": 0,
        "text_generated_slot_logit_mean": 0.,
        "text_generated_slot_logit_std": 0.,
        "text_generated_slot_logit_min": 0.,
        "text_generated_slot_logit_max": 0.,
    }
    result.update(diagnostics)
    return result


class nnUNetTrainerSynapseReconstructionScreen(nnUNetTrainerSynapseMasked):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.screen_arm = os.environ["SYNAPSE_SCREEN_ARM"]
        if self.screen_arm not in ARMS:
            raise ValueError(self.screen_arm)
        if self.is_ddp:
            raise ValueError("This initial screen is single-GPU only")
        self.reconstruction_weight = .25 if self.screen_arm == "reconstruction" else 0.
        self.query_every = 4
        self._screen_iteration = 0
        self._screen_diagnostics = []

    def on_train_epoch_start(self):
        self._screen_iteration = 0
        self._screen_diagnostics = []
        super().on_train_epoch_start()

    def use_all_queries(self):
        step = self.current_epoch * self.num_iterations_per_epoch + self._screen_iteration
        return self.screen_arm in ("all_query_control", "reconstruction") and step % self.query_every == 0

    def joint_loss(self, data, target, checkpoint_decoder=True):
        """Cache encoder features; recompute decoder/loss activations on backward.

        Both all-query arms use identical query frequency and average (not sum)
        the five structured losses. No auxiliary learned head is introduced.
        """
        network = self._unwrap_network()
        with self._autocast_context():
            skips = network.encode(data)
        probabilities, structured = {}, []
        for g in range(self.num_dynamic_groups):
            # Bind group as a default: checkpoint recomputes AFTER this loop.
            def query(*features, group=g):
                ids = torch.full((data.shape[0],), group, dtype=torch.long, device=data.device)
                remapped, valid, active = self._remap_target_for_group(target, ids)
                with self._autocast_context():
                    output = network.decode_from_skips(list(features), ids)
                output = self._output_to_float(output)
                loss = self._compute_structured_loss(output, remapped, valid, active)
                return masked_probabilities(get_main_output(output), group), loss

            if checkpoint_decoder:
                p, loss = checkpoint(query, *skips, use_reentrant=False, preserve_rng_state=True)
            else:
                p, loss = query(*skips)
            probabilities[g] = p
            structured.append(loss)
        structured_loss = torch.stack(structured).mean()
        high_target = target[0] if isinstance(target, list) else target
        # Compute the readout in BOTH arms; only its coefficient differs.
        reconstruction = original_label_loss(reconstruct_scores(probabilities), high_target)
        total = structured_loss + self.reconstruction_weight * reconstruction
        return total, structured_loss, reconstruction

    def train_step(self, batch):
        joint = self.use_all_queries()
        self._screen_iteration += 1
        if not joint:
            return screen_step_output(**super().train_step(batch))
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._to_device_target(batch["target"], self.device)
        high = target[0] if isinstance(target, list) else target
        for g, spec in enumerate(DYNAMIC_GROUP_SPECS):
            self._group_sample_counter_epoch[g] += data.shape[0]
            for sample in high:
                voxels = sum(int((sample == label).sum()) for label in spec.original_labels)
                self._query_coverage[g] += (1, int(voxels > 0), voxels)
        self.network.zero_grad(set_to_none=True)
        loss, structured, reconstruction = self.joint_loss(data, target)
        self._assert_all_finite(loss, "joint reconstruction loss")
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
        else:
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self._get_optimizer_parameters(), self.grad_clip_norm)
        if not self._all_finite(norm):
            raise FloatingPointError("Nonfinite reconstruction gradient")
        if self.grad_scaler is not None:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()
        self._screen_diagnostics.append([float(structured.detach()), float(reconstruction.detach()), float(norm)])
        # Epoch collation requires the same schema as inherited single-query
        # steps, regardless of which query type happens to start the epoch.
        # This screen has no text-generated slot head, so these are inactive.
        return screen_step_output(loss.detach().cpu().numpy(), structured.detach().cpu().numpy())

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        if self._screen_diagnostics:
            values = np.asarray(self._screen_diagnostics)
            self.print_to_log_file("[Reconstruction screen]", {
                "all_query_updates": len(values), "structured_loss": float(values[:, 0].mean()),
                "original_label_loss": float(values[:, 1].mean()), "gradient_norm": float(values[:, 2].mean()),
                "reconstruction_weight": self.reconstruction_weight,
            })
