"""CPU contract tests for the four AnchorSlot training-side ablations.

A query-aware foreground sampling, B Dice without empty-GT entries, C region-balanced suppression,
D grouped-softmax flat head. Every option defaults to the legacy behaviour.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.variants.structured_conditional import (
    dataloader_structured_conditional_er_dynamic as dl_mod,
)
from nnunetv2.training.nnUNetTrainer.variants.structured_conditional import (
    grouped_softmax_flat_er_dynamic as gsf,
)
from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.label_mapping_er_dynamic import (
    BACKGROUND_CHANNEL,
    COND_SLOT_1_CHANNEL,
    COND_SLOT_3_CHANNEL,
    FIXED_OUTPUT_TO_ORIGINAL,
    NUM_DYNAMIC_GROUPS,
    NUM_OUTPUT_CHANNELS,
    ORIGINAL_LABEL_TO_GROUP_ID,
    OTHER_CHANNEL,
    get_group_spec,
    remap_original_to_structured,
)
from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.structured_loss_er_dynamic import (
    StructuredConditionalLoss,
    StructuredLossConfig,
)

SPATIAL = (4, 5, 3)
MITO, GOLGI, PEROX = 0, 1, 11


def _batch(seed: int = 0, groups=(MITO, GOLGI)):
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn((len(groups), NUM_OUTPUT_CHANNELS, *SPATIAL), generator=gen, requires_grad=True)
    seg = torch.randint(0, 32, (len(groups), 1, *SPATIAL), generator=gen)
    target, valid, active = remap_original_to_structured(seg, torch.tensor(groups))
    return logits, target, valid, active


# --- B: Dice without empty-GT entries --------------------------------------------------------


def _legacy_per_sample_dice(logits, target, valid_mask, class_mask, smooth=1e-5):
    """Verbatim copy of the pre-change non-batch Dice path (backward-compatibility reference)."""
    probs = torch.softmax(logits, dim=1)
    onehot = torch.nn.functional.one_hot(target[:, 0].long(), NUM_OUTPUT_CHANNELS)
    onehot = onehot.permute(0, -1, *range(1, onehot.ndim - 1)).float()
    cm = class_mask.float().view(class_mask.shape[0], class_mask.shape[1], *([1] * (logits.ndim - 2)))
    probs, onehot, valid = probs * cm, onehot * cm, valid_mask.float()
    axes = tuple(range(2, logits.ndim))
    intersect = (probs * onehot * valid).sum(dim=axes)
    pred_sum = (probs * valid).sum(dim=axes)
    gt_sum = (onehot * valid).sum(dim=axes)
    dice = (2.0 * intersect + smooth) / (pred_sum + gt_sum + smooth).clamp_min(1e-8)
    return 1.0 - dice[class_mask].mean()


def test_default_dice_is_legacy():
    logits, target, valid, active = _batch()
    loss = StructuredConditionalLoss(StructuredLossConfig(batch_dice=False))
    cm = loss._build_foreground_class_mask(active, logits.device)
    assert torch.allclose(loss._masked_soft_dice(logits, target, valid, cm),
                          _legacy_per_sample_dice(logits, target, valid, cm), atol=0, rtol=0)


def test_dice_non_empty_ignores_classes_absent_from_the_sample():
    logits, target, valid, active = _batch(seed=1)
    target = target.clone()
    target[target == COND_SLOT_1_CHANNEL] = BACKGROUND_CHANNEL  # slot 1 now has empty GT in both samples
    cfg = StructuredLossConfig(batch_dice=False, dice_exclude_empty_gt=True)
    loss = StructuredConditionalLoss(cfg)
    cm = loss._build_foreground_class_mask(active, logits.device)
    got = loss._masked_soft_dice(logits, target, valid, cm)
    onehot = torch.nn.functional.one_hot(target[:, 0], NUM_OUTPUT_CHANNELS).movedim(-1, 1)
    non_empty = cm & (onehot.flatten(2).sum(-1) > 0)
    assert not non_empty[:, COND_SLOT_1_CHANNEL].any()
    assert torch.allclose(got, _legacy_per_sample_dice(logits, target, valid, non_empty))


def test_dice_non_empty_returns_connected_zero_when_every_entry_is_empty():
    logits, target, valid, active = _batch(seed=2)
    empty = torch.zeros_like(target)  # background everywhere; background is not a Dice class
    loss = StructuredConditionalLoss(StructuredLossConfig(batch_dice=False, dice_exclude_empty_gt=True))
    cm = loss._build_foreground_class_mask(active, logits.device)
    out = loss._masked_soft_dice(logits, empty, valid, cm)
    assert float(out) == 0.0
    out.backward()
    assert logits.grad is not None


def test_dice_non_empty_batch_dice_path():
    logits, target, valid, active = _batch(seed=3)
    target = target.clone()
    target[target == OTHER_CHANNEL] = BACKGROUND_CHANNEL
    on = StructuredConditionalLoss(StructuredLossConfig(batch_dice=True, dice_exclude_empty_gt=True))
    off = StructuredConditionalLoss(StructuredLossConfig(batch_dice=True))
    cm = on._build_foreground_class_mask(active, logits.device)
    assert float(on._masked_soft_dice(logits, target, valid, cm)) < float(off._masked_soft_dice(logits, target, valid, cm))


# --- C: region-balanced suppression ----------------------------------------------------------


def _suppression_case():
    # 1 sample, golgi query (slots 1-2 active). 16 bg voxels, 2 anchor voxels, 2 "other" voxels, 0 slot voxels.
    target = torch.zeros((1, 1, 20), dtype=torch.long)
    target[0, 0, 16:18] = 3  # cyto anchor
    target[0, 0, 18:20] = OTHER_CHANNEL
    valid = torch.ones_like(target, dtype=torch.bool)
    active = torch.tensor([[True, True, False]])
    logits = torch.full((1, NUM_OUTPUT_CHANNELS, 20), -20.0)
    logits[0, BACKGROUND_CHANNEL, :16] = 20.0
    logits[0, 3, 16:18] = 20.0
    logits[0, COND_SLOT_1_CHANNEL, 18:20] = 20.0  # leakage onto non-query organelle only
    return logits, target, valid, active


def test_default_suppression_is_voxel_pooled():
    logits, target, valid, active = _suppression_case()
    loss = StructuredConditionalLoss(StructuredLossConfig())
    # slot-1 prob ~1 on 2 of 20 voxels, 2 active slots -> (2 * 1) / (2 * 20)
    assert float(loss._suppression_loss(logits, target, valid, active)) == pytest.approx(0.05, abs=1e-4)


def test_region_balanced_suppression_averages_regions():
    logits, target, valid, active = _suppression_case()
    loss = StructuredConditionalLoss(StructuredLossConfig(suppression_mode="region_balanced"))
    # regions bg 0, anchors 0, other (2*1)/(2*2)=0.5 -> mean 0.5/3
    assert float(loss._suppression_loss(logits, target, valid, active)) == pytest.approx(0.5 / 3, abs=1e-4)


def test_region_balanced_suppression_skips_empty_regions():
    logits, target, valid, active = _suppression_case()
    target = target.clone()
    target[0, 0, 16:18] = 0  # no anchor voxels left
    loss = StructuredConditionalLoss(StructuredLossConfig(suppression_mode="region_balanced"))
    assert float(loss._suppression_loss(logits, target, valid, active)) == pytest.approx(0.5 / 2, abs=1e-4)


def test_region_balanced_suppression_ignores_invalid_voxels():
    logits, target, valid, active = _suppression_case()
    valid = valid.clone()
    valid[0, 0, 18:20] = False
    loss = StructuredConditionalLoss(StructuredLossConfig(suppression_mode="region_balanced"))
    assert float(loss._suppression_loss(logits, target, valid, active)) == pytest.approx(0.0, abs=1e-4)


def test_unknown_suppression_mode_raises():
    with pytest.raises(ValueError, match="suppression_mode"):
        StructuredConditionalLoss(StructuredLossConfig(suppression_mode="median"))


# --- A: query-aware foreground sampling ------------------------------------------------------


def _locs(n: int) -> np.ndarray:
    return np.zeros((n, 4), dtype=np.int64)


def test_select_query_foreground_class_picks_only_present_group_labels():
    class_locations = {4: _locs(3), 5: _locs(0), 6: _locs(2), 7: _locs(5)}
    rng = np.random.default_rng(0)
    picks = {dl_mod.select_query_foreground_class(class_locations, MITO, rng) for _ in range(200)}
    assert picks == {4, 6}


def test_select_query_foreground_class_returns_none_for_absent_group():
    class_locations = {4: _locs(3), 30: _locs(0)}
    assert dl_mod.select_query_foreground_class(class_locations, PEROX, np.random.default_rng(0)) is None
    assert dl_mod.select_query_foreground_class({}, PEROX, np.random.default_rng(0)) is None


class _FakeData:
    def __init__(self, class_locations):
        self.class_locations = class_locations

    def load_case(self, key):
        data = np.zeros((1, 4, 4, 4), dtype=np.float32)
        seg = np.zeros((1, 4, 4, 4), dtype=np.int16)
        return data, seg, {"class_locations": self.class_locations}


def _fake_loader(query_aware: bool, force_fg: bool, class_locations):
    loader = dl_mod.StructuredConditionalDataLoader3D.__new__(dl_mod.StructuredConditionalDataLoader3D)
    loader.batch_size = 1
    loader.data_shape = (1, 1, 4, 4, 4)
    loader.seg_shape = (1, 1, 4, 4, 4)
    loader.rng = np.random.default_rng(0)
    loader.p_present_group = 1.0
    loader.group_sampling_mode = "random"
    loader._case_present_group_cache = {}
    loader.transforms = None
    loader.query_aware_foreground = query_aware
    loader._data = _FakeData(class_locations)
    loader.get_indices = lambda: ["case"]
    loader.get_do_oversample = lambda j: force_fg
    calls = []

    def get_bbox(shape, fg, locs, overwrite_class=None, verbose=False):
        calls.append({"force_fg": fg, "overwrite_class": overwrite_class})
        return [0, 0, 0], [4, 4, 4]

    loader.get_bbox = get_bbox
    return loader, calls


def test_query_aware_loader_centres_force_fg_samples_on_the_queried_group():
    class_locations = {30: _locs(4), 4: _locs(9)}
    loader, calls = _fake_loader(query_aware=True, force_fg=True, class_locations={30: class_locations[30]})
    batch = loader.generate_train_batch()
    assert int(batch["group_id"][0]) == PEROX
    assert calls == [{"force_fg": True, "overwrite_class": 30}]


def test_query_aware_loader_leaves_random_samples_untouched():
    loader, calls = _fake_loader(query_aware=True, force_fg=False, class_locations={30: _locs(4)})
    loader.generate_train_batch()
    assert calls == [{"force_fg": False, "overwrite_class": None}]


def test_default_loader_never_overrides_the_class():
    loader, calls = _fake_loader(query_aware=False, force_fg=True, class_locations={30: _locs(4)})
    loader.generate_train_batch()
    assert calls == [{"force_fg": True, "overwrite_class": None}]


# --- D: grouped-softmax flat head ------------------------------------------------------------


def test_flat_layout_covers_every_label_once():
    channels = [gsf.FLAT_CHANNEL_OF_ORIGINAL_LABEL[label] for label in range(32)]
    assert sorted(channels) == list(range(32))
    assert gsf.NUM_FLAT_OUTPUT_CHANNELS == 32 + NUM_DYNAMIC_GROUPS
    others = {gsf.OTHER_FLAT_CHANNEL[g] for g in range(NUM_DYNAMIC_GROUPS)}
    assert others == set(range(32, 32 + NUM_DYNAMIC_GROUPS))


def test_gather_group_logits_maps_channels_per_sample():
    flat = torch.arange(gsf.NUM_FLAT_OUTPUT_CHANNELS, dtype=torch.float32).view(1, -1, 1).repeat(2, 1, 3)
    out = gsf.gather_group_logits(flat, torch.tensor([MITO, GOLGI]))
    assert out.shape == (2, NUM_OUTPUT_CHANNELS, 3)
    for b, g in enumerate((MITO, GOLGI)):
        spec = get_group_spec(g)
        assert float(out[b, BACKGROUND_CHANNEL, 0]) == gsf.FLAT_CHANNEL_OF_ORIGINAL_LABEL[0]
        for channel, original in FIXED_OUTPUT_TO_ORIGINAL.items():
            assert float(out[b, channel, 0]) == gsf.FLAT_CHANNEL_OF_ORIGINAL_LABEL[original]
        for k, original in enumerate(spec.original_labels):
            assert float(out[b, COND_SLOT_1_CHANNEL + k, 0]) == gsf.FLAT_CHANNEL_OF_ORIGINAL_LABEL[original]
        if spec.num_slots < 3:
            assert torch.all(out[b, COND_SLOT_3_CHANNEL] == gsf.INACTIVE_SLOT_LOGIT)
        assert float(out[b, OTHER_CHANNEL, 0]) == gsf.OTHER_FLAT_CHANNEL[g]


def test_gather_is_differentiable_and_ignores_padding():
    flat = torch.randn((1, gsf.NUM_FLAT_OUTPUT_CHANNELS, 2), requires_grad=True)
    gsf.gather_group_logits(flat, torch.tensor([GOLGI])).sum().backward()
    used = {gsf.FLAT_CHANNEL_OF_ORIGINAL_LABEL[x] for x in (0, 1, 2, 3, 27, 7, 8)} | {gsf.OTHER_FLAT_CHANNEL[GOLGI]}
    assert {int(c) for c in torch.nonzero(flat.grad[0, :, 0]).flatten()} == used


class _FakeDecoder:
    def __init__(self, deep_supervision):
        self.deep_supervision = deep_supervision


class _FakeBackbone(nn.Module):
    def __init__(self, deep_supervision: bool):
        super().__init__()
        self.decoder = _FakeDecoder(deep_supervision)
        self.proj = nn.Conv3d(1, gsf.NUM_FLAT_OUTPUT_CHANNELS, 1)

    def forward(self, x):
        y = self.proj(x)
        return [y, y[:, :, ::2, ::2, ::2]] if self.decoder.deep_supervision else y


@pytest.mark.parametrize("deep_supervision", [False, True])
def test_flat_net_forward_matches_gather_and_encode_decode(deep_supervision):
    net = gsf.GroupedSoftmaxFlatNet(_FakeBackbone(deep_supervision))
    x = torch.randn((2, 1, 4, 4, 4))
    ids = torch.tensor([PEROX, MITO])
    flat = net(x, all_groups=True)
    per_group = net(x, ids)
    via_skips = net.decode_from_skips(net.encode(x), ids)
    if deep_supervision:
        assert isinstance(per_group, list) and len(per_group) == 2
        expected = [gsf.gather_group_logits(f, ids) for f in flat]
        for a, b, c in zip(per_group, expected, via_skips):
            assert torch.equal(a, b) and torch.equal(a, c)
    else:
        assert torch.equal(per_group, gsf.gather_group_logits(flat, ids))
        assert torch.equal(via_skips, per_group)
    assert net.decoder is net.backbone.decoder


def _loss_pieces():
    loss = StructuredConditionalLoss(StructuredLossConfig(batch_dice=False))

    def compute(outputs, target, valid, active):
        return loss(outputs, target, valid, active)

    def remap(target, group_ids):
        return remap_original_to_structured(target, group_ids)

    return compute, remap


@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_grouped_loss_is_mean_of_per_group_structured_losses(use_checkpoint):
    compute, remap = _loss_pieces()
    gen = torch.Generator().manual_seed(5)
    flat = torch.randn((2, gsf.NUM_FLAT_OUTPUT_CHANNELS, *SPATIAL), generator=gen, requires_grad=True)
    seg = torch.randint(0, 32, (2, 1, *SPATIAL), generator=gen)
    got = gsf.grouped_softmax_loss(flat, seg, compute, remap, use_checkpoint=use_checkpoint)
    expected = torch.stack([
        compute(gsf.gather_group_logits(flat, torch.full((2,), g)), *remap(seg, torch.full((2,), g)))
        for g in range(NUM_DYNAMIC_GROUPS)
    ]).mean()
    assert torch.allclose(got, expected, atol=1e-6)


def test_grouped_loss_checkpointing_gives_identical_gradients():
    compute, remap = _loss_pieces()
    gen = torch.Generator().manual_seed(6)
    base = torch.randn((1, gsf.NUM_FLAT_OUTPUT_CHANNELS, *SPATIAL), generator=gen)
    seg = torch.randint(0, 32, (1, 1, *SPATIAL), generator=gen)
    grads = []
    for use_checkpoint in (False, True):
        flat = base.clone().requires_grad_(True)
        gsf.grouped_softmax_loss(flat, seg, compute, remap, use_checkpoint=use_checkpoint).backward()
        grads.append(flat.grad)
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def test_grouped_loss_supports_deep_supervision_lists():
    loss = StructuredConditionalLoss(StructuredLossConfig(batch_dice=False))
    weights = [0.75, 0.25]

    def compute(outputs, targets, valids, active):
        return sum(w * loss(o, t, v, active) for w, o, t, v in zip(weights, outputs, targets, valids))

    def remap(targets, group_ids):
        pieces = [remap_original_to_structured(t, group_ids) for t in targets]
        return [p[0] for p in pieces], [p[1] for p in pieces], pieces[0][2]

    gen = torch.Generator().manual_seed(7)
    flat = [torch.randn((1, gsf.NUM_FLAT_OUTPUT_CHANNELS, 4, 4, 4), generator=gen, requires_grad=True),
            torch.randn((1, gsf.NUM_FLAT_OUTPUT_CHANNELS, 2, 2, 2), generator=gen, requires_grad=True)]
    seg = [torch.randint(0, 32, (1, 1, 4, 4, 4), generator=gen), torch.randint(0, 32, (1, 1, 2, 2, 2), generator=gen)]
    out = gsf.grouped_softmax_loss(flat, seg, compute, remap, use_checkpoint=True)
    out.backward()
    assert torch.isfinite(out) and flat[0].grad is not None and flat[1].grad is not None


# --- trainer wiring (real Dataset204 plans, CPU, results redirected to tmp) --------------------

PREPROCESSED = Path("/projects/weilab/liupeng/data/nnUNet/nnUNet_preprocessed/"
                    "Dataset204_Dataset101_CellMap_model_high_res_hdrfix_all")


@pytest.fixture
def trainer_factory(tmp_path, monkeypatch):
    if not (PREPROCESSED / "nnUNetPlans.json").is_file():
        pytest.skip("Dataset204 preprocessed plans not available")
    import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as base_module
    from nnunetv2.training.nnUNetTrainer.variants.structured_conditional import (
        trainer_structured_conditional_er_dynamic_ablations as ablations,
    )

    monkeypatch.setattr(base_module, "nnUNet_results", str(tmp_path))
    plans = json.loads((PREPROCESSED / "nnUNetPlans.json").read_text())
    dataset_json = json.loads((PREPROCESSED / "dataset.json").read_text())

    def make(name: str):
        cls = getattr(ablations, name) if name != "base" else ablations.nnUNetTrainerStructuredConditionalERDynamic
        return cls(plans, "3d_lowres_large_patch", 0, dataset_json, device=torch.device("cpu"))

    return make


def test_base_trainer_defaults_are_legacy(trainer_factory):
    t = trainer_factory("base")
    assert t.loss_cfg.dice_exclude_empty_gt is False
    assert t.loss_cfg.suppression_mode == "pooled"
    assert getattr(t, "query_aware_foreground", False) is False


@pytest.mark.parametrize(
    "name,check",
    [
        ("nnUNetTrainerStructuredConditionalERDynamicQueryAwareFG", lambda t: t.query_aware_foreground is True),
        ("nnUNetTrainerStructuredConditionalERDynamicDiceNonEmpty", lambda t: t.loss_cfg.dice_exclude_empty_gt),
        ("nnUNetTrainerStructuredConditionalERDynamicRegionSuppress",
         lambda t: t.loss_cfg.suppression_mode == "region_balanced"),
    ],
)
def test_ablation_trainers_change_exactly_one_option(trainer_factory, name, check):
    base, t = trainer_factory("base"), trainer_factory(name)
    assert check(t)
    changed = {f for f in vars(base.loss_cfg) if getattr(base.loss_cfg, f) != getattr(t.loss_cfg, f)}
    assert len(changed) <= 1
    assert getattr(t, "query_aware_foreground", False) == (name.endswith("QueryAwareFG"))


def test_grouped_trainer_builds_flat_network_with_legacy_loss(trainer_factory):
    from nnunetv2.training.nnUNetTrainer.variants.structured_conditional import (
        trainer_structured_conditional_er_dynamic_ablations as ablations,
    )

    t = trainer_factory("nnUNetTrainerStructuredConditionalERDynamicGroupedSoftmaxFlat")
    assert replace(t.loss_cfg) == replace(trainer_factory("base").loss_cfg)
    arch = t.configuration_manager.network_arch_class_name
    net = ablations.nnUNetTrainerStructuredConditionalERDynamicGroupedSoftmaxFlat.build_network_architecture(
        arch, t.configuration_manager.network_arch_init_kwargs,
        t.configuration_manager.network_arch_init_kwargs_req_import, 1, 99, enable_deep_supervision=True,
    )
    assert isinstance(net, gsf.GroupedSoftmaxFlatNet)
    assert net.backbone.decoder.seg_layers[-1].out_channels == gsf.NUM_FLAT_OUTPUT_CHANNELS


# --- D: table-based remapping must equal the reference remap exactly ---------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_remap_equals_reference_for_every_group(seed):
    gen = torch.Generator().manual_seed(seed)
    seg = torch.randint(-1, 36, (3, 1, *SPATIAL), generator=gen)  # includes padding (-1) and unknown ids
    for g in range(NUM_DYNAMIC_GROUPS):
        ids = torch.tensor([g, (g + 5) % NUM_DYNAMIC_GROUPS, (g + 9) % NUM_DYNAMIC_GROUPS])
        ref = remap_original_to_structured(seg, ids)
        fast = gsf.remap_original_to_structured_fast(seg, ids)
        for a, b in zip(ref, fast):
            assert a.dtype == b.dtype and torch.equal(a, b)


def test_fast_remap_honours_ignore_label_and_multichannel_input():
    seg = torch.tensor([[[0, 4, 7, 30, 2]], [[5, 5, -1, 31, 30]]]).repeat(1, 2, 1)  # 2 channels, keep first
    ids = torch.tensor([PEROX, MITO])
    ref = remap_original_to_structured(seg, ids, ignore_label=30)
    fast = gsf.remap_original_to_structured_fast(seg, ids, ignore_label=30)
    for a, b in zip(ref, fast):
        assert torch.equal(a, b)


def test_fast_remap_target_handles_deep_supervision_lists():
    gen = torch.Generator().manual_seed(9)
    targets = [torch.randint(-1, 32, (2, 1, 4, 4, 4), generator=gen), torch.randint(-1, 32, (2, 1, 2, 2, 2), generator=gen)]
    ids = torch.tensor([MITO, GOLGI])
    remapped, valid, active = gsf.remap_target_fast(targets, ids)
    for t, r, v in zip(targets, remapped, valid):
        ref = remap_original_to_structured(t, ids)
        assert torch.equal(r, ref[0]) and torch.equal(v, ref[1])
    assert torch.equal(active, remap_original_to_structured(targets[0], ids)[2])


# --- D1: flat head with the base one-query-per-sample training step ----------------------------


def test_flat_head_sampled_trainer_uses_base_train_step_and_flat_network(trainer_factory):
    from nnunetv2.training.nnUNetTrainer.variants.structured_conditional import (
        trainer_structured_conditional_er_dynamic_ablations as ablations,
    )

    d1 = ablations.nnUNetTrainerStructuredConditionalERDynamicFlatHeadSampled
    d13 = ablations.nnUNetTrainerStructuredConditionalERDynamicGroupedSoftmaxFlat
    assert d1.train_step is ablations.nnUNetTrainerStructuredConditionalERDynamic.train_step
    assert d13.train_step is not d1.train_step
    assert issubclass(d13, d1)
    t = trainer_factory("nnUNetTrainerStructuredConditionalERDynamicFlatHeadSampled")
    assert replace(t.loss_cfg) == replace(trainer_factory("base").loss_cfg)
    net = d1.build_network_architecture(
        t.configuration_manager.network_arch_class_name, t.configuration_manager.network_arch_init_kwargs,
        t.configuration_manager.network_arch_init_kwargs_req_import, 1, 99, enable_deep_supervision=False,
    )
    assert isinstance(net, gsf.GroupedSoftmaxFlatNet)
    with torch.no_grad():
        out = net.eval()(torch.zeros((1, 1, 64, 64, 64)), torch.tensor([PEROX]))  # 6 stages need >= 64^3
    assert out.shape == (1, NUM_OUTPUT_CHANNELS, 64, 64, 64)
