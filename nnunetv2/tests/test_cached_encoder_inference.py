"""Encoder-cached all-group inference (--cache_encoder) must reproduce the per-group sliding-window logits exactly.

CPU only, tiny network.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from torch import nn

from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.network_structured_conditional import (
    StructuredConditionalUNet,
)

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "predict_structured_conditional.py"
_spec = importlib.util.spec_from_file_location("predict_structured_conditional_cached_test", _SCRIPT)
psc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(psc)
group_chunks, groups_per_pass = psc.group_chunks, psc.groups_per_pass

NUM_GROUPS = 13
PATCH = (16, 16, 16)


def _tiny_network(seed: int = 0) -> StructuredConditionalUNet:
    torch.manual_seed(seed)
    backbone = PlainConvUNet(
        input_channels=1, n_stages=3, features_per_stage=[4, 8, 16], conv_op=nn.Conv3d, kernel_sizes=3,
        strides=[1, 2, 2], n_conv_per_stage=2, num_classes=9, n_conv_per_stage_decoder=2, conv_bias=True,
        norm_op=nn.InstanceNorm3d, norm_op_kwargs={"eps": 1e-5, "affine": True}, nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True}, deep_supervision=False,
    )
    net = StructuredConditionalUNet(backbone, num_groups=NUM_GROUPS, num_output_channels=9, cond_dim=8)
    # FiLM starts as the identity; randomize it so that different groups give different outputs.
    for film in net.decoder_film:
        nn.init.normal_(film.affine.weight, std=0.5)
        nn.init.normal_(film.affine.bias, std=0.5)
    return net.eval()


def _predictor(net: nn.Module, use_mirroring: bool = True, max_groups_per_pass=None):
    p = psc.StructuredConditionalPredictor(
        group_id=0, run_all_groups=True, output_mode="original", fixed_merge_mode="mean",
        tile_step_size=0.5, use_gaussian=True, use_mirroring=use_mirroring, perform_everything_on_device=False,
        device=torch.device("cpu"), allow_tqdm=False, cache_encoder=True, max_groups_per_pass=max_groups_per_pass,
    )
    p.network = net
    p.list_of_parameters = [net.state_dict()]
    p.configuration_manager = SimpleNamespace(patch_size=list(PATCH))
    p.allowed_mirroring_axes = (0, 1, 2)
    p.num_dynamic_groups = NUM_GROUPS
    p.is_hierarchical_parallel = False
    return p


def _per_group_reference(p, image: torch.Tensor) -> dict:
    reference = {}
    for g in range(NUM_GROUPS):
        p.group_id = g
        reference[g] = p.predict_logits_from_preprocessed_data(image)
    return reference


@pytest.fixture
def image() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(1, 20, 27, 30)  # not a multiple of the patch or step, so tiles overlap unevenly


@pytest.mark.parametrize("use_mirroring", [True, False])
def test_cached_logits_equal_per_group_logits(image, use_mirroring):
    p = _predictor(_tiny_network(), use_mirroring=use_mirroring)
    reference = _per_group_reference(p, image)
    cached = p.predict_logits_all_groups_from_preprocessed_data(image)
    assert sorted(cached) == list(range(NUM_GROUPS))
    for g in range(NUM_GROUPS):
        assert cached[g].shape == reference[g].shape == (9, *image.shape[1:])
        assert torch.equal(cached[g], reference[g]), f"group {g}: max diff {(cached[g] - reference[g]).abs().max()}"


def test_groups_give_different_logits(image):
    """Guards the test above: identical groups would hide a group mix-up."""
    cached = _predictor(_tiny_network()).predict_logits_all_groups_from_preprocessed_data(image)
    assert not torch.allclose(cached[0], cached[5])


def test_group_chunking_does_not_change_logits(image):
    net = _tiny_network()
    full = _predictor(net).predict_logits_all_groups_from_preprocessed_data(image)
    chunked = _predictor(net, max_groups_per_pass=4).predict_logits_all_groups_from_preprocessed_data(image)
    for g in range(NUM_GROUPS):
        assert torch.equal(full[g], chunked[g])


def test_encoder_runs_once_per_tile_and_mirror_per_chunk(image, monkeypatch):
    net = _tiny_network()
    p = _predictor(net, max_groups_per_pass=5)  # 13 groups -> chunks of 5, 5, 3
    calls = {"encode": 0, "decode": 0}
    encode, decode = net.encode, net.decode_from_skips

    def counting_encode(x):
        calls["encode"] += 1
        return encode(x)

    def counting_decode(skips, group_ids, **kwargs):
        calls["decode"] += 1
        return decode(skips, group_ids, **kwargs)

    monkeypatch.setattr(net, "encode", counting_encode)
    monkeypatch.setattr(net, "decode_from_skips", counting_decode)
    p.predict_logits_all_groups_from_preprocessed_data(image)
    padded = p._pad_for_sliding_window(image)[0]
    n_tiles = len(p._internal_get_sliding_window_slicers(padded.shape[1:]))
    assert calls["encode"] == n_tiles * 8 * 3
    assert calls["decode"] == n_tiles * 8 * NUM_GROUPS


def test_group_chunks_cover_all_groups_in_order():
    assert group_chunks(13, 13) == [list(range(13))]
    assert group_chunks(13, 5) == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12]]
    assert group_chunks(3, 1) == [[0], [1], [2]]
    with pytest.raises(ValueError):
        group_chunks(13, 0)


def test_groups_per_pass_from_free_memory():
    gib = 1024 ** 3
    # 400^3 volume, 9 channels: ~2.4 GiB of fp32 accumulators per group.
    assert groups_per_pass(free_bytes=44 * gib, num_voxels=400 ** 3, patch_voxels=160 ** 3, num_channels=9,
                           num_groups=13) == 13
    # 512^3 volume does not fit all 13 groups on a 48 GB card.
    k = groups_per_pass(free_bytes=44 * gib, num_voxels=512 ** 3, patch_voxels=160 ** 3, num_channels=9,
                        num_groups=13)
    assert 1 <= k < 13
    assert groups_per_pass(free_bytes=1 * gib, num_voxels=512 ** 3, patch_voxels=160 ** 3, num_channels=9,
                           num_groups=13) == 1


def test_runtime_error_falls_back_like_released_predictor(image, monkeypatch):
    """A non-OOM failure (e.g. non-finite AMP logits) retries without mirroring, as the per-group predictor does."""
    net = _tiny_network()
    p = _predictor(net, use_mirroring=True)
    original = p._internal_predict_sliding_window_groups
    calls = {"n": 0}

    def fail_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Conditioned network forward produced non-finite logits.")
        return original(*args, **kwargs)

    monkeypatch.setattr(p, "_internal_predict_sliding_window_groups", fail_once)
    cached = p.predict_logits_all_groups_from_preprocessed_data(image)
    assert p.use_mirroring  # restored after the fallback
    reference = _per_group_reference(_predictor(net, use_mirroring=False), image)
    for g in range(NUM_GROUPS):
        assert torch.equal(cached[g], reference[g])
