"""Tests for the AnchorSlot group-classification head (2b): coarse targets, loss, network modes, merges."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nnunetv2.training.nnUNetTrainer.variants.structured_conditional import (
    trainer_structured_conditional_er_dynamic_group_head as gh,
)
from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.label_mapping_er_dynamic import (
    DYNAMIC_GROUP_SPECS,
    FIXED_OUTPUT_TO_ORIGINAL,
)

PREPROCESSED = Path("/projects/weilab/liupeng/data/nnUNet/nnUNet_preprocessed/"
                    "Dataset205_Dataset101_CellMap_model_high_res_truegeom")


# --- coarse label mapping -----------------------------------------------------------------------

def test_group_lut_covers_every_original_label():
    lut = gh.build_group_lut()
    assert lut.shape == (32,) and gh.NUM_GROUP_HEAD_CLASSES == 5 + len(DYNAMIC_GROUP_SPECS) == 18
    assert int(lut[0]) == 0
    for i, (_, label) in enumerate(sorted(FIXED_OUTPUT_TO_ORIGINAL.items())):
        assert int(lut[int(label)]) == 1 + i
    for spec in DYNAMIC_GROUP_SPECS:
        for label in spec.original_labels:
            assert int(lut[int(label)]) == 5 + spec.group_id
    assert set(lut.tolist()) == set(range(gh.NUM_GROUP_HEAD_CLASSES))


def test_group_targets_ignore_negative_labels_and_invalid_voxels():
    target = torch.tensor([[[[[0, 1, 4, -1]]]]], dtype=torch.float32)       # B=1, C=1, 1x1x4
    valid = torch.tensor([[[[[True, True, False, True]]]]])
    coarse = gh.group_targets(target, valid, gh.build_group_lut())
    assert coarse.tolist() == [[[[0, 1, gh.IGNORE_INDEX, gh.IGNORE_INDEX]]]]


# --- loss -------------------------------------------------------------------------------------------

def _rand_case(seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn((2, gh.NUM_GROUP_HEAD_CLASSES, 4, 4, 4), generator=g)
    target = torch.randint(0, 32, (2, 1, 4, 4, 4), generator=g).float()
    valid = torch.rand((2, 1, 4, 4, 4), generator=g) > 0.3
    return logits, target, valid


def test_group_loss_is_finite_and_ignores_invalid_voxels():
    logits, target, valid = _rand_case()
    lut = gh.build_group_lut()
    loss_a = gh.group_head_loss(logits, target, valid, lut)
    logits_b, target_b = logits.clone(), target.clone()
    logits_b[valid.expand_as(logits_b) == 0] = 100.0                          # change only invalid voxels
    target_b[~valid] = 7
    loss_b = gh.group_head_loss(logits_b, target_b, valid, lut)
    assert torch.isfinite(loss_a) and torch.allclose(loss_a, loss_b, atol=1e-6)


def test_group_loss_prefers_correct_predictions():
    _, target, valid = _rand_case(1)
    lut = gh.build_group_lut()
    coarse = gh.group_targets(target, valid, lut)
    good = torch.full((2, gh.NUM_GROUP_HEAD_CLASSES, 4, 4, 4), -10.0)
    good.scatter_(1, coarse.clamp(min=0).unsqueeze(1), 10.0)
    bad = torch.zeros_like(good)
    assert gh.group_head_loss(good, target, valid, lut) < gh.group_head_loss(bad, target, valid, lut)


def test_group_loss_all_invalid_returns_connected_zero():
    logits, target, _ = _rand_case(2)
    logits.requires_grad_(True)
    loss = gh.group_head_loss(logits, target, torch.zeros_like(target, dtype=torch.bool), gh.build_group_lut())
    loss.backward()
    assert float(loss) == 0.0 and logits.grad is not None


# --- network built from the real plans ----------------------------------------------------------------

@pytest.fixture
def plans_cm():
    if not (PREPROCESSED / "nnUNetPlans.json").is_file():
        pytest.skip("Dataset205 preprocessed plans not available")
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    pm = PlansManager(json.loads((PREPROCESSED / "nnUNetPlans.json").read_text()))
    return pm.get_configuration("3d_lowres_large_patch")


def _build(cm, append=False, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("NNUNET_STRUCTCOND_APPEND_GROUP_LOGITS", "1" if append else "0")
    return gh.nnUNetTrainerStructuredConditionalERDynamicGroupHead.build_network_architecture(
        cm.network_arch_class_name, cm.network_arch_init_kwargs, cm.network_arch_init_kwargs_req_import,
        1, 99, enable_deep_supervision=True)


def test_group_head_leaves_main_output_unchanged_and_adds_modes(plans_cm, monkeypatch):
    torch.manual_seed(0)
    net = _build(plans_cm, append=False, monkeypatch=monkeypatch).eval()
    assert isinstance(net, gh.StructuredConditionalUNetGroupHead)
    assert net.group_head.out_channels == gh.NUM_GROUP_HEAD_CLASSES
    x = torch.randn((1, 1, 64, 64, 64)); ids = torch.tensor([4])
    with torch.no_grad():
        plain = net(x, ids)
        pack = net(x, ids, return_group_logits=True)
        net.append_group_logits_to_main = True
        appended = net(x, ids)
    assert isinstance(plain, list) and plain[0].shape[1] == 9
    assert torch.equal(pack["logits"][0], plain[0])
    assert pack["group_logits"].shape == (1, gh.NUM_GROUP_HEAD_CLASSES, *plain[0].shape[2:])
    assert appended[0].shape[1] == 9 + gh.NUM_GROUP_HEAD_CLASSES
    assert torch.equal(appended[0][:, :9], plain[0]) and torch.equal(appended[0][:, 9:], pack["group_logits"])
    assert all(torch.equal(a, b) for a, b in zip(appended[1:], plain[1:]))


def test_group_head_env_flag_sets_append_mode(plans_cm, monkeypatch):
    assert _build(plans_cm, append=True, monkeypatch=monkeypatch).append_group_logits_to_main is True
    assert _build(plans_cm, append=False, monkeypatch=monkeypatch).append_group_logits_to_main is False


def test_group_loss_backpropagates_into_head_and_decoder(plans_cm, monkeypatch):
    net = _build(plans_cm, monkeypatch=monkeypatch).train()
    x = torch.randn((1, 1, 64, 64, 64)); ids = torch.tensor([3])
    target = torch.randint(0, 32, (1, 1, 64, 64, 64)).float()
    pack = net(x, ids, return_group_logits=True)
    loss = gh.group_head_loss(pack["group_logits"], target, torch.ones_like(target, dtype=torch.bool), gh.build_group_lut())
    loss.backward()
    assert net.group_head.weight.grad is not None and net.group_head.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.decoder.stages[-1].parameters())


# --- merges from compact probabilities ----------------------------------------------------------------

def _toy_compact():
    """Two groups (endo=3, lyso=4) competing on one voxel; channels follow the dump layout."""
    meta = [dict(index=0, group=-1, kind="background_mean", label=0)]
    meta += [dict(index=1 + i, group=-1, kind="anchor_mean", label=l) for i, l in enumerate([1, 2, 3, 27])]
    rows = {0: 0.05, 1: 0.01, 2: 0.01, 3: 0.05, 4: 0.01}
    for g, labels, slots, other in [(3, [11, 12], [0.10, 0.55], 0.30), (4, [13, 14], [0.08, 0.50], 0.05)]:
        for l, s in zip(labels, slots):
            rows[len(meta)] = s; meta.append(dict(index=len(meta), group=g, kind="slot", label=l))
        rows[len(meta)] = other; meta.append(dict(index=len(meta), group=g, kind="other", label=-1))
    p = np.array([rows[i] for i in range(len(meta))], dtype=np.float32)[:, None, None, None]
    return p, meta


def test_legacy_scores_pick_highest_slot_and_mix_zero_is_legacy():
    p, meta = _toy_compact()
    head = np.zeros((gh.NUM_GROUP_HEAD_CLASSES, 1, 1, 1), np.float32); head[5 + 4] = 1.0
    legacy = gh.compact_scores(p, meta, head, mode="legacy")
    assert int(legacy.argmax(0).ravel()[0]) == 12                       # endo_lum 0.55 beats lyso_lum 0.50
    assert np.array_equal(gh.compact_scores(p, meta, head, mode="mix", weight=0.0).argmax(0), legacy.argmax(0))


@pytest.mark.parametrize("mode,weight", [("arbiter", 1.0), ("mix", 1.0), ("mix", 0.5)])
def test_group_head_arbitrates_between_groups(mode, weight):
    p, meta = _toy_compact()
    head = np.full((gh.NUM_GROUP_HEAD_CLASSES, 1, 1, 1), 0.01, np.float32); head[5 + 4] = 0.8   # head says lysosome
    out = gh.compact_scores(p, meta, head, mode=mode, weight=weight)
    assert int(out.argmax(0).ravel()[0]) == 14                          # lyso_lum


def test_arbiter_keeps_the_legacy_dynamic_versus_anchor_decision():
    p, meta = _toy_compact()
    p[3] = 0.9                                                           # cytoplasm now wins in legacy
    head = np.full((gh.NUM_GROUP_HEAD_CLASSES, 1, 1, 1), 0.01, np.float32); head[5 + 4] = 0.9
    assert int(gh.compact_scores(p, meta, head, mode="arbiter", weight=1.0).argmax(0).ravel()[0]) == 3


# --- trainer wiring (real Dataset205 plans, CPU, results redirected to tmp) ---------------------------

def test_group_head_trainer_keeps_the_base_recipe(tmp_path, monkeypatch):
    if not (PREPROCESSED / "nnUNetPlans.json").is_file():
        pytest.skip("Dataset205 preprocessed plans not available")
    from dataclasses import replace
    import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as base_module
    from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.trainer_structured_conditional_er_dynamic import (
        nnUNetTrainerStructuredConditionalERDynamic as Base)
    monkeypatch.setattr(base_module, "nnUNet_results", str(tmp_path))
    monkeypatch.delenv("NNUNET_STRUCTCOND_LAMBDA_GROUP", raising=False)
    plans = json.loads((PREPROCESSED / "nnUNetPlans.json").read_text())
    dsj = json.loads((PREPROCESSED / "dataset.json").read_text())
    t = gh.nnUNetTrainerStructuredConditionalERDynamicGroupHead(plans, "3d_lowres_large_patch", 0, dsj, device=torch.device("cpu"))
    b = Base(plans, "3d_lowres_large_patch", 0, dsj, device=torch.device("cpu"))
    assert t.lambda_group == 0.5
    assert replace(t.loss_cfg) == replace(b.loss_cfg)
    assert torch.equal(t.group_lut, gh.build_group_lut())
