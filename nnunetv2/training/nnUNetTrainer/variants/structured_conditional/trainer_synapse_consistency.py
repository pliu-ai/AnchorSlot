"""Separately named, checkpoint-compatible Synapse correction ablations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .label_mapping_synapse import DYNAMIC_GROUP_SPECS
from .synapse_consistency import ConsistentSynapseLoss, QueryAwareSynapseDataLoader, mask_group_logits
from .trainer_structured_conditional_synapse import nnUNetTrainerStructuredConditionalSynapse


class nnUNetTrainerSynapseLegacyControl(nnUNetTrainerStructuredConditionalSynapse):
    """Legacy objective and sampling, with the same coverage logging as new arms."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self._query_coverage = np.zeros((self.num_dynamic_groups, 3), dtype=np.int64)

    def train_step(self, batch):
        target = batch["target"][0] if isinstance(batch["target"], list) else batch["target"]
        ids = batch.get("group_id")
        if ids is None:
            raise RuntimeError("Synapse consistency screen requires the 3D query dataloader")
        for index, group in enumerate(torch.as_tensor(ids).tolist()):
            voxels = sum(int((target[index] == label).sum()) for label in DYNAMIC_GROUP_SPECS[group].original_labels)
            self._query_coverage[group] += (1, int(voxels > 0), voxels)
        return super().train_step(batch)

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        counts = self._query_coverage.copy()
        if self.is_ddp:
            counts = self._ddp_sum_array(counts)
        report = {spec.display_name: {
            "queries": int(counts[g, 0]), "positive_queries": int(counts[g, 1]),
            "query_voxels": int(counts[g, 2]),
            "positive_fraction": float(counts[g, 1] / max(1, counts[g, 0])),
        } for g, spec in enumerate(DYNAMIC_GROUP_SPECS)}
        if self.local_rank == 0:
            with (Path(self.output_folder) / "query_coverage.jsonl").open("a") as handle:
                handle.write(json.dumps({"epoch": int(self.current_epoch), "after_augmentation": True,
                                         "groups": report}) + "\n")
        self.print_to_log_file("[Synapse query coverage] " + json.dumps(report))
        self._query_coverage[:] = 0


class nnUNetTrainerSynapseMasked(nnUNetTrainerSynapseLegacyControl):
    conditional_sample_dice = False

    def _build_loss(self):
        return ConsistentSynapseLoss(self.loss_cfg, self.conditional_sample_dice)

    def prepare_group_logits(self, logits, group_id):
        return mask_group_logits(logits, group_id)


class nnUNetTrainerSynapseMaskedQueryCrop(nnUNetTrainerSynapseMasked):
    structured_dataloader_class = QueryAwareSynapseDataLoader


class nnUNetTrainerSynapseMaskedSampleDice(nnUNetTrainerSynapseMasked):
    conditional_sample_dice = True
