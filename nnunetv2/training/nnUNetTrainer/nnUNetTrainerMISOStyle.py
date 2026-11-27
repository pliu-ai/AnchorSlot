"""Flat sigmoid baseline with the training ingredients of MISO (Kuruba et al., bioRxiv 2026).

Everything else follows the Dataset205 flat arm (nnUNetTrainerRalph: same backbone, patches, 4-GPU schedule,
learning rate, epochs, and checkpoint rule). The differences are MISO's three ingredients:
  * output: one sigmoid per original label (31 regions); inference takes the argmax over labels and background
    where no probability exceeds 0.5 (ArgmaxRegionLabelManager);
  * loss: Tversky (alpha=0.3, beta=0.7) + BCE (MaskedTverskyBCELoss), with deep supervision as in nnU-Net;
  * masking: labels that a training crop does not annotate (CellMap train_crop_manifest.csv) are ignored;
  * sampling: repeat-factor crop sampling, r_i = max(1, max_c sqrt(t / f_c)) over labels c present in crop i,
    where f_c is the share of training crops containing c (MISO does not report t; MISO_REPEAT_T, default 0.5).
MISO's entropy-based pixel exclusion is not reproduced.

Environment: MISO_STATS_JSON (required) = train_label_stats.json with labels.*.crop_fraction,
present_labels_per_crop and annotated_labels_per_crop; MISO_REPEAT_T (optional).
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Mapping, Sequence

import numpy as np
import torch

import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as base_trainer_module
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.tversky_bce import MaskedTverskyBCELoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerRalph import nnUNetTrainerRalph


def to_region_dataset_json(dataset_json: Mapping) -> dict:
    """One region per foreground label, ordered by label id; idempotent for an already converted dataset.json."""
    labels = dataset_json["labels"]

    def label_id(value) -> int:
        return int(value[0]) if isinstance(value, (list, tuple)) else int(value)

    foreground = sorted((label_id(v), k) for k, v in labels.items() if k != "background")
    converted = deepcopy(dict(dataset_json))
    converted["labels"] = {"background": 0, **{name: [lid] for lid, name in foreground}}
    converted["regions_class_order"] = [lid for lid, _ in foreground]
    return converted


def repeat_factors(keys: Sequence[str], present_by_crop: Mapping[str, Sequence[str]],
                   crop_fraction: Mapping[str, float], t: float) -> np.ndarray:
    """LVIS-style repeat factor per crop from its rarest present label (1 if nothing present is rarer than t)."""
    factors = []
    for key in keys:
        per_label = [max(1.0, float(np.sqrt(t / crop_fraction[c])))
                     for c in present_by_crop.get(key, ()) if crop_fraction.get(c, 0.0) > 0.0]
        factors.append(max(per_label) if per_label else 1.0)
    return np.asarray(factors, dtype=np.float64)


class nnUNetTrainerMISOStyle(nnUNetTrainerRalph):
    TVERSKY_ALPHA = 0.3
    TVERSKY_BETA = 0.7

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device("cuda")):
        plans = deepcopy(plans)
        plans["label_manager"] = "ArgmaxRegionLabelManager"
        super().__init__(plans, configuration, fold, to_region_dataset_json(dataset_json), unpack_dataset, device)
        stats_path = os.environ.get("MISO_STATS_JSON")
        if not stats_path or not os.path.isfile(stats_path):
            raise RuntimeError("nnUNetTrainerMISOStyle needs MISO_STATS_JSON pointing to train_label_stats.json")
        with open(stats_path) as f:
            stats = json.load(f)
        self.repeat_t = float(os.environ.get("MISO_REPEAT_T", "0.5"))
        self.crop_fraction = {k: float(v["crop_fraction"]) for k, v in stats["labels"].items()}
        self.present_by_crop = {k: list(v) for k, v in stats["present_labels_per_crop"].items()}
        self.annotated_by_crop = {k: set(v) for k, v in stats["annotated_labels_per_crop"].items()}
        id_to_name = {v[0]: k for k, v in self.dataset_json["labels"].items() if k != "background"}
        self.channel_labels = [id_to_name[i] for i in self.dataset_json["regions_class_order"]]
        missing = set(self.channel_labels) - set(self.crop_fraction)
        if missing:
            raise RuntimeError(f"MISO_STATS_JSON lacks labels {sorted(missing)}")

    def _build_loss(self):
        loss = MaskedTverskyBCELoss(alpha=self.TVERSKY_ALPHA, beta=self.TVERSKY_BETA)
        if self.enable_deep_supervision:
            weights = np.array([1 / (2 ** i) for i in range(len(self._get_deep_supervision_scales()))])
            # Same weighting as nnU-Net; DDP without compile needs a tiny non-zero weight on the lowest output.
            weights[-1] = 1e-6 if (self.is_ddp and not self._do_i_compile()) else 0
            loss = DeepSupervisionWrapper(loss, weights / weights.sum())
        return loss

    # --- repeat-factor sampling ------------------------------------------------------------------------------------
    def training_sampling_probabilities(self, keys: Sequence[str]) -> np.ndarray | None:
        """Probabilities for the training loader; None for any other key set (e.g. validation crops)."""
        keys = list(keys)
        if not keys or not all(k in self.present_by_crop for k in keys):
            return None
        factors = repeat_factors(keys, self.present_by_crop, self.crop_fraction, self.repeat_t)
        return factors / factors.sum()

    def get_dataloaders(self):
        trainer = self
        original = base_trainer_module.nnUNetDataLoader3D

        def loader_with_repeat_factors(data, *args, **kwargs):
            if kwargs.get("sampling_probabilities") is None:
                p = trainer.training_sampling_probabilities(list(data.keys()))
                if p is not None:
                    trainer.print_to_log_file(f"MISO repeat-factor sampling: t={trainer.repeat_t}, {len(p)} crops, "
                                              f"max/min probability ratio {p.max() / p.min():.2f}")
                kwargs["sampling_probabilities"] = p
            return original(data, *args, **kwargs)

        # nnU-Net builds and starts the loaders inside get_dataloaders; swap the constructor only for that call.
        base_trainer_module.nnUNetDataLoader3D = loader_with_repeat_factors
        try:
            return super().get_dataloaders()
        finally:
            base_trainer_module.nnUNetDataLoader3D = original

    # --- masking of unannotated labels -----------------------------------------------------------------------------
    def annotation_mask(self, keys: Sequence[str]) -> torch.Tensor | None:
        """(B, C) bool, True where the crop annotates the channel's label; None if every label is annotated."""
        mask = torch.ones((len(keys), len(self.channel_labels)), dtype=torch.bool)
        for b, key in enumerate(keys):
            annotated = self.annotated_by_crop.get(key)
            if annotated is None:
                continue
            for c, name in enumerate(self.channel_labels):
                mask[b, c] = name in annotated
        return None if bool(mask.all()) else mask

    @staticmethod
    def mask_target(target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        view = mask.view(*mask.shape, *([1] * (target.ndim - 2)))
        return target.to(torch.int16).masked_fill(~view.to(target.device), -1)

    def train_step(self, batch: dict) -> dict:
        mask = self.annotation_mask(batch["keys"])
        if mask is not None:
            batch = dict(batch)
            target = batch["target"]
            batch["target"] = ([self.mask_target(t, mask) for t in target] if isinstance(target, list)
                               else self.mask_target(target, mask))
        return super().train_step(batch)
