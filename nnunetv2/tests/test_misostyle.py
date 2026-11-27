"""Tests for the MISO-style flat baseline (loss, label manager, sampling, masking, trainer wiring)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from nnunetv2.training.loss.tversky_bce import MaskedTverskyBCELoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMISOStyle import repeat_factors, to_region_dataset_json
from nnunetv2.utilities.label_handling.label_handling import get_labelmanager_class_from_plans
from nnunetv2.utilities.label_handling.label_handling_argmax import ArgmaxRegionLabelManager

RUN = Path(__file__).resolve().parents[3]
STATS = RUN / "miso_stats.json"
PREPROCESSED = Path("/projects/weilab/liupeng/data/nnUNet/nnUNet_preprocessed/"
                    "Dataset205_Dataset101_CellMap_model_high_res_truegeom")


# --- loss ------------------------------------------------------------------------------------------------------------
def test_masked_channels_do_not_change_loss_or_gradient():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 4, 4, 4, requires_grad=True)
    target = (torch.rand(2, 3, 4, 4, 4) > 0.7).float()
    masked = target.clone()
    masked[0, 1] = -1
    loss_fn = MaskedTverskyBCELoss()
    loss_masked = loss_fn(logits, masked)
    loss_masked.backward()
    assert torch.all(logits.grad[0, 1] == 0)
    # Changing the ignored channel's logits must not change the loss.
    other = logits.detach().clone()
    other[0, 1] = 100.0
    assert torch.isclose(loss_fn(other, masked), loss_masked.detach())


def test_perfect_prediction_has_near_zero_tversky_term():
    target = torch.zeros(1, 2, 8, 8, 8)
    target[0, 0, :4] = 1
    target[0, 1, 4:] = 1
    logits = (target * 2 - 1) * 20
    loss = MaskedTverskyBCELoss()(logits, target)
    assert loss.item() < 1e-3


def test_misses_cost_more_than_false_alarms():
    target = torch.zeros(1, 1, 10, 10, 10)
    target[0, 0, :5] = 1
    miss = target.clone()
    miss[0, 0, :2] = 0      # 200 false negatives
    alarm = target.clone()
    alarm[0, 0, 5:7] = 1    # 200 false positives
    loss_fn = MaskedTverskyBCELoss(alpha=0.3, beta=0.7)
    assert loss_fn((miss * 2 - 1) * 20, target) > loss_fn((alarm * 2 - 1) * 20, target)


# --- label manager -------------------------------------------------------------------------------------------------
def _manager():
    labels = {"background": 0, "a": [1], "b": [2], "c": [5]}
    return ArgmaxRegionLabelManager(label_dict=labels, regions_class_order=[1, 2, 5])


def test_plans_select_argmax_label_manager():
    assert get_labelmanager_class_from_plans({"label_manager": "ArgmaxRegionLabelManager"}) is ArgmaxRegionLabelManager


@pytest.mark.parametrize("as_torch", [False, True])
def test_argmax_segmentation_with_background_threshold(as_torch):
    probs = np.zeros((3, 1, 1, 4), dtype=np.float32)
    probs[:, 0, 0, 0] = [0.9, 0.95, 0.1]   # both a and b above 0.5 -> argmax b (label 2)
    probs[:, 0, 0, 1] = [0.2, 0.3, 0.4]    # none above 0.5 -> background
    probs[:, 0, 0, 2] = [0.1, 0.2, 0.7]    # c -> label 5
    probs[:, 0, 0, 3] = [0.6, 0.1, 0.1]    # a -> label 1
    x = torch.from_numpy(probs) if as_torch else probs
    seg = _manager().convert_probabilities_to_segmentation(x)
    seg = seg.cpu().numpy() if as_torch else seg
    assert seg.ravel().tolist() == [2, 0, 5, 1]


# --- dataset.json and sampling -------------------------------------------------------------------------------------
def test_region_dataset_json_is_ordered_and_idempotent():
    ds = {"labels": {"background": 0, "b": 2, "a": 1}, "file_ending": ".nii.gz"}
    once = to_region_dataset_json(ds)
    assert once["labels"] == {"background": 0, "a": [1], "b": [2]}
    assert once["regions_class_order"] == [1, 2]
    assert to_region_dataset_json(once) == once
    assert ds["labels"]["a"] == 1  # input not modified


def test_repeat_factors_follow_rarest_present_label():
    f = repeat_factors(["x", "y", "z"], {"x": ["rare", "common"], "y": ["common"], "z": []},
                       {"rare": 0.02, "common": 0.9}, t=0.5)
    assert f[0] == pytest.approx(5.0)
    assert f[1] == 1.0 and f[2] == 1.0


# --- trainer wiring on the real Dataset205 plans (CPU) ---------------------------------------------------------------
@pytest.fixture
def trainer(tmp_path, monkeypatch):
    if not (PREPROCESSED / "nnUNetPlans.json").is_file() or not STATS.is_file():
        pytest.skip("Dataset205 plans or miso_stats.json not available")
    import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as base_module
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMISOStyle import nnUNetTrainerMISOStyle

    monkeypatch.setattr(base_module, "nnUNet_results", str(tmp_path))
    monkeypatch.setenv("MISO_STATS_JSON", str(STATS))
    monkeypatch.delenv("RALPH_CONFIG_JSON", raising=False)
    plans = json.loads((PREPROCESSED / "nnUNetPlans.json").read_text())
    dataset_json = json.loads((PREPROCESSED / "dataset.json").read_text())
    return nnUNetTrainerMISOStyle(plans, "3d_lowres_large_patch_bs4", 0, dataset_json, device=torch.device("cpu"))


def test_trainer_uses_31_sigmoid_regions(trainer):
    assert isinstance(trainer.label_manager, ArgmaxRegionLabelManager)
    assert trainer.label_manager.has_regions
    assert trainer.label_manager.num_segmentation_heads == 31
    assert trainer.channel_labels[0] == "ecs" and trainer.channel_labels[-1] == "perox_lum"


def test_trainer_sampling_and_masking(trainer):
    stats = json.loads(STATS.read_text())
    train_keys = list(stats["present_labels_per_crop"])
    p = trainer.training_sampling_probabilities(train_keys)
    assert p is not None and p.shape == (202,) and p.sum() == pytest.approx(1.0) and p.max() > p.min()
    assert trainer.training_sampling_probabilities(["jrc_sum159-4_crop219"]) is None  # validation crop
    partial = [k for k, v in stats["annotated_labels_per_crop"].items() if len(v) < 31]
    full = [k for k, v in stats["annotated_labels_per_crop"].items() if len(v) == 31][:2]
    assert trainer.annotation_mask(full) is None
    mask = trainer.annotation_mask(partial)
    assert mask is not None and (~mask).sum().item() == sum(31 - len(stats["annotated_labels_per_crop"][k]) for k in partial)
    target = torch.ones(len(partial), 31, 2, 2, 2)
    masked = trainer.mask_target(target, mask)
    assert (masked == -1).sum().item() == (~mask).sum().item() * 8


def test_trainer_loss_is_deep_supervised(trainer):
    from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

    loss = trainer._build_loss()
    assert isinstance(loss, DeepSupervisionWrapper)
    assert isinstance(loss.loss, MaskedTverskyBCELoss)
