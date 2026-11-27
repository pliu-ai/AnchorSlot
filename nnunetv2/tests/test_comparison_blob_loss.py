"""Numerical and semantic checks for the flat component-balanced baseline."""
from __future__ import annotations

from datetime import timedelta
import gc
import socket
import time

import pytest
from scipy import ndimage
import torch

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.nnUNetTrainer.variants.comparison_baselines import blob_loss as module
from nnunetv2.training.nnUNetTrainer.variants.comparison_baselines.blob_loss import (
    FlatCEAndDiceAndBlobLoss,
    MulticlassBlobDiceLoss,
    prepare_blob_components,
)


def dense_reference(logits, target, smooth=1e-5, valid=None, instance_ids=None):
    """Slow independent oracle: explicitly mask all voxels for each component."""
    p = logits.softmax(1)
    valid = target >= 0 if valid is None else valid & (target >= 0)
    images = []
    for b in range(target.shape[0]):
        classes = []
        for c in range(1, logits.shape[1]):
            positive = (target[b] == c) & valid[b]
            if not positive.any():
                continue
            if instance_ids is None:
                labels, _ = ndimage.label(positive.cpu().numpy())
                labels = torch.from_numpy(labels).to(logits.device)
            else:
                labels = torch.where(positive, instance_ids[b], 0)
            instances = []
            for k in labels.unique():
                if k == 0:
                    continue
                own = labels == k
                domain = valid[b] & (~positive | own)
                numerator = 2 * p[b, c][own].sum() + smooth
                denominator = p[b, c][domain].sum() + own.sum() + smooth
                instances.append(1 - numerator / denominator.clamp_min(1e-8))
            classes.append(torch.stack(instances).mean())
        if classes:
            images.append(torch.stack(classes).mean())
    return torch.stack(images).mean() if images else logits.reshape(-1)[:0].sum()


@pytest.mark.parametrize("spatial", [(5, 7), (3, 4, 5)])
@pytest.mark.parametrize("channel_target", [False, True])
@pytest.mark.parametrize("backend", ["auto", "scipy"])
def test_scatter_matches_dense_blob_values_and_gradients(spatial, channel_target, backend, monkeypatch):
    if backend == "scipy":
        monkeypatch.setattr(module, "cc3d", None)
    torch.manual_seed(42)
    target = torch.randint(-1, 4, (3, *spatial))
    target[1] = 0  # Empty images are skipped by blob, not counted as easy blobs.
    valid = torch.rand(target.shape) > .12
    logits = torch.randn(3, 4, *spatial, dtype=torch.double, requires_grad=True)
    actual = MulticlassBlobDiceLoss()(logits, target[:, None] if channel_target else target, valid_mask=valid)
    expected = dense_reference(logits, target, valid=valid)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    ga = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    ge = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(ga, ge, atol=1e-12, rtol=1e-12)
    assert torch.count_nonzero(ga.movedim(1, -1)[~valid | (target < 0)]) == 0


def test_each_blob_excludes_other_same_class_components_but_retains_all_background():
    # Class 1 has two isolated blobs. Class 2 is one-vs-rest background for 1.
    target = torch.tensor([[[1, 0, 1, 0, 2]]])
    probabilities = torch.tensor([[[[.1, .7, .2, .8, .1]],
                                   [[.8, .1, .7, .1, .2]],
                                   [[.1, .2, .1, .1, .7]]]], dtype=torch.double)
    logits = probabilities.log().requires_grad_()
    actual = MulticlassBlobDiceLoss(smooth=0)(logits, target)
    class1 = ((1 - 1.6 / (.8 + 1 + .4)) + (1 - 1.4 / (.7 + 1 + .4))) / 2
    class2 = 1 - 1.4 / (.7 + 1 + .5)
    torch.testing.assert_close(actual, torch.tensor((class1 + class2) / 2, dtype=torch.double))
    # Penalize a false positive even far from every GT component (no ROI crop).
    changed = probabilities.clone()
    changed[0, :, 0, 3] = torch.tensor([.2, .7, .1])
    assert MulticlassBlobDiceLoss(smooth=0)(changed.log(), target) > actual


@pytest.mark.parametrize("batch_dice", [False, True])
@pytest.mark.parametrize("smooth", [0., 1e-5, 1.])
def test_blob_weight_zero_matches_nnunet_global_loss_and_gradients(batch_dice, smooth):
    torch.manual_seed(101)
    logits = torch.randn(2, 32, 4, 5, dtype=torch.double, requires_grad=True)
    target = torch.randint(0, 32, (2, 1, 4, 5))
    target[0, 0, 1, 1] = -1
    actual = FlatCEAndDiceAndBlobLoss(weight_blob=0, batch_dice=batch_dice, smooth=smooth)(logits, target)
    reference = DC_and_CE_loss(
        dict(batch_dice=batch_dice, smooth=smooth, do_bg=False, ddp=False), {}, ignore_label=-1
    )
    expected = reference(logits, target)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    ga = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    ge = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(ga, ge, atol=1e-12, rtol=1e-12)


def test_compound_loss_weights_and_prepared_cpu_components():
    torch.manual_seed(43)
    logits = torch.randn(2, 4, 5, 6, dtype=torch.double, requires_grad=True)
    target = torch.randint(0, 4, (2, 1, 5, 6))
    target[0, 0, 0] = -100
    components = prepare_blob_components(target, 4)
    loss = FlatCEAndDiceAndBlobLoss(weight_ce=.7, weight_dice=.4, weight_blob=1.3)
    expected = FlatCEAndDiceAndBlobLoss(weight_ce=.7, weight_dice=.4, weight_blob=0)(logits, target)
    expected += 1.3 * dense_reference(logits, target[:, 0])
    actual = loss(logits, target, components=components)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, loss(logits, target))


def test_provided_instance_ids_split_touching_objects_preserve_disconnected_fragments():
    target = torch.tensor([[[1, 1, 0, 1, 2, 2]]])
    # The ID 101 in class 1 has two fragments; it does not merge with class 2.
    ids = torch.tensor([[[101, 2**50, 0, 101, 101, 101]]])
    components = prepare_blob_components(target, 3, instance_ids=ids)
    assert components.class_ids.tolist() == [1, 1, 2]
    assert components.sizes.tolist() == [2, 1, 2]
    assert components.labels[0, 0, 0] == components.labels[0, 0, 3]
    assert components.labels[0, 0, 0] != components.labels[0, 0, 1]
    logits = torch.randn(1, 3, 1, 6, dtype=torch.double, requires_grad=True)
    loss = MulticlassBlobDiceLoss()(logits, target, instance_ids=ids)
    torch.testing.assert_close(loss, dense_reference(logits, target, instance_ids=ids))


def test_component_ids_are_namespaced_across_images():
    target = torch.ones(2, 1, 2, dtype=torch.long)
    ids = torch.full_like(target, 123)
    components = prepare_blob_components(target, 2, instance_ids=ids)
    assert components.class_ids.tolist() == [1, 1]
    assert components.batch_ids.tolist() == [0, 1]
    assert components.labels[0, 0, 0] != components.labels[1, 0, 0]


@pytest.mark.parametrize("backend", ["auto", "scipy"])
@pytest.mark.parametrize("dimensions", [2, 3])
def test_connectivity_is_explicit_and_does_not_merge_classes(backend, dimensions, monkeypatch):
    if backend == "scipy":
        monkeypatch.setattr(module, "cc3d", None)
    target = torch.zeros((1, *([2] * dimensions)), dtype=torch.long)
    target[(0, *([0] * dimensions))] = 1
    target[(0, *([1] * dimensions))] = 1
    face = prepare_blob_components(target, 3, connectivity=1, spatial_dims=dimensions)
    full = prepare_blob_components(target, 3, connectivity=dimensions, spatial_dims=dimensions)
    assert face.sizes.tolist() == [1, 1]
    assert full.sizes.tolist() == [2]
    target[(0, *([1] * dimensions))] = 2
    full = prepare_blob_components(target, 3, connectivity=dimensions, spatial_dims=dimensions)
    assert sorted(full.class_ids.tolist()) == [1, 2]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch_dice", [False, True])
def test_all_ignored_half_precision_is_finite_differentiable_zero(dtype, batch_dice):
    logits = torch.full((2, 32, 8, 9), 30000., dtype=dtype, requires_grad=True)
    target = torch.full((2, 1, 8, 9), -1, dtype=torch.long)
    logits.data[0, :, 0, 0] = float("nan")  # Ignored voxels cannot poison the loss.
    loss = FlatCEAndDiceAndBlobLoss(batch_dice=batch_dice)(logits, target)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_autocast_does_not_reduce_blob_or_dice_accumulation_precision(dtype):
    torch.manual_seed(51)
    logits = torch.randn(2, 5, 4, 5, dtype=dtype, requires_grad=True)
    target = torch.randint(0, 5, (2, 4, 5))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = FlatCEAndDiceAndBlobLoss()(logits, target)
    expected = FlatCEAndDiceAndBlobLoss()(logits.float(), target)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()


def test_empty_foreground_has_zero_blob_but_global_loss_penalizes_false_positives():
    target = torch.zeros(1, 1, 3, 4, dtype=torch.long)
    logits = torch.zeros(1, 3, 3, 4, requires_grad=True)
    blob = MulticlassBlobDiceLoss()(logits, target)
    assert blob.item() == 0 and blob.requires_grad
    total = FlatCEAndDiceAndBlobLoss()(logits, target)
    total.backward()
    assert torch.all(logits.grad[:, 1:] > 0)
    assert torch.all(logits.grad[:, 0] < 0)


def test_missing_classes_do_not_dilute_blob_class_average():
    target = torch.tensor([[[1, 0, 1, 0, 1, 0, 2]]])
    logits = torch.randn(1, 8, 1, 7, dtype=torch.double)
    # Oracle averages class 1's three blobs, then gives class 2 the same weight.
    torch.testing.assert_close(MulticlassBlobDiceLoss()(logits, target), dense_reference(logits, target))


def test_missing_instance_ids_and_wrong_precomputed_maps_fail_explicitly():
    target = torch.tensor([[[0, 1, 1, 0]]])
    with pytest.raises(ValueError, match="cover all"):
        prepare_blob_components(target, 2, instance_ids=torch.zeros_like(target))
    components = prepare_blob_components(target, 2)
    logits = torch.randn(1, 2, 1, 4)
    changed = torch.tensor([[[0, 1, 0, 1]]])
    with pytest.raises(ValueError, match="exact augmented"):
        MulticlassBlobDiceLoss()(logits, changed, components=components)
    with pytest.raises(ValueError, match="either"):
        MulticlassBlobDiceLoss()(logits, target, components=components, instance_ids=target)
    with pytest.raises(ValueError, match="shape or class"):
        MulticlassBlobDiceLoss()(torch.randn(1, 3, 1, 4), target, components=components)


def test_ignore_label_and_explicit_validity_remove_voxels_from_all_terms():
    target = torch.tensor([[[1, 255, 900, -100, 0, 1]]])
    valid = torch.tensor([[[True, True, False, True, True, True]]])
    logits = torch.randn(1, 3, 1, 6, dtype=torch.double, requires_grad=True)
    actual = FlatCEAndDiceAndBlobLoss(ignore_label=255)(logits, target, valid_mask=valid)
    cropped = logits[:, :, :, [0, 4, 5]]
    # Explicit IDs preserve the original two blobs even when cropping joins them.
    expected = FlatCEAndDiceAndBlobLoss()(cropped, torch.tensor([[[1, 0, 1]]]))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.count_nonzero(logits.grad[:, :, :, 1:4]) == 0


def test_rejects_nonintegral_or_out_of_range_valid_semantics():
    logits = torch.randn(1, 3, 1, 3)
    for target in (torch.tensor([[[0., .5, 1.]]]), torch.tensor([[[0., float("nan"), 1.]]]),
                   torch.tensor([[[0, 3, 1]]])):
        with pytest.raises(ValueError):
            FlatCEAndDiceAndBlobLoss()(logits, target)


def test_default_preparation_and_explicit_spatial_dims_for_depth_one():
    target = torch.zeros(1, 1, 3, 4, dtype=torch.long)  # channel-less 3D depth=1
    target[0, 0, 1, 1] = 1
    components = prepare_blob_components(target, 2, spatial_dims=3)
    assert tuple(components.labels.shape) == (1, 1, 3, 4)
    logits = torch.randn(1, 2, 1, 3, 4)
    torch.testing.assert_close(MulticlassBlobDiceLoss()(logits, target, components=components),
                               MulticlassBlobDiceLoss()(logits, target[:, None]))


def test_blob_disabled_skips_connected_component_work(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Connected components should not run for the CE+Dice control.")
    monkeypatch.setattr(module, "prepare_blob_components", unexpected)
    loss = FlatCEAndDiceAndBlobLoss(weight_blob=0)(torch.randn(1, 32, 3, 4), torch.zeros(1, 1, 3, 4).long())
    assert torch.isfinite(loss)


def test_many_blobs_use_single_component_map_and_match_dense_oracle():
    target = torch.zeros(1, 15, 15, dtype=torch.long)
    target[:, ::2, ::2] = 1
    components = prepare_blob_components(target, 2)
    assert len(components.sizes) == 64
    assert components.labels.numel() == target.numel()
    assert components.sizes.numel() == 64
    logits = torch.randn(1, 2, 15, 15, dtype=torch.double)
    torch.testing.assert_close(MulticlassBlobDiceLoss()(logits, target, components=components),
                               dense_reference(logits, target))


def test_tiny_blob_keeps_small_background_mass_next_to_a_large_blob():
    target = torch.ones(1, 1, 65539, dtype=torch.long)
    target[0, 0, -3] = 0
    target[0, 0, -1] = 0
    logits = torch.zeros(1, 2, 1, 65539)
    logits[:, 1] = 1000.  # Exact unit probabilities on a large foreground blob.
    logits[0, 1, 0, -3] = logits[0, 1, 0, -1] = -6.906754778648554
    logits[0, 1, 0, -2] = 0.
    actual = MulticlassBlobDiceLoss(smooth=0)(logits, target)
    expected = dense_reference(logits.double(), target, smooth=0)
    torch.testing.assert_close(actual.double(), expected, atol=1e-7, rtol=1e-7)


def _distributed_dice_worker(rank, init_method, empty_second_rank):
    import torch.distributed as dist
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=2,
        timeout=timedelta(seconds=60),
    )
    # Keep the backend alive when the generic ProcessGroup is destroyed. In
    # torch 2.5.1, the direct ProcessGroupGloo Python holder releases the GIL
    # during destruction, whereas generic ProcessGroup destruction does not:
    # https://github.com/pytorch/pytorch/blob/v2.5.1/torch/csrc/distributed/c10d/init.cpp
    # Releasing the final reference through that holder avoids the GIL/worker
    # destructor deadlock observed in combined test runs.
    backend = dist.group.WORLD._get_backend(torch.device("cpu"))
    try:
        torch.manual_seed(77)
        full_logits = torch.randn(2, 4, 3, 4, dtype=torch.double, requires_grad=True)
        target = torch.randint(0, 4, (2, 1, 3, 4))
        if empty_second_rank:
            target[1] = -100
        local = full_logits.detach()[rank:rank + 1].clone().requires_grad_()
        actual = FlatCEAndDiceAndBlobLoss(
            weight_ce=0, weight_blob=0, batch_dice=True, ddp=True, ignore_label=None
        )(local, target[rank:rank + 1])
        expected = FlatCEAndDiceAndBlobLoss(
            weight_ce=0, weight_blob=0, batch_dice=True, ddp=False, ignore_label=None
        )(full_logits, target)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        expected.backward()
        # nnU-Net's differentiable all-gather sums both replicated losses;
        # DDP subsequently averages parameter gradients over its two ranks.
        torch.testing.assert_close(local.grad / 2, full_logits.grad[rank:rank + 1])
        del actual, expected, local, full_logits, target
        gc.collect()
        # Drain autograd all-reduce work before process-group destruction.
        # Otherwise a Gloo work thread can need the GIL to release a Tensor
        # while the destructor holds the GIL and waits for that thread to exit.
        dist.barrier()
    finally:
        try:
            dist.destroy_process_group()
        finally:
            del backend


@pytest.mark.parametrize("empty_second_rank", [False, True])
def test_distributed_batch_dice_matches_global_batch_even_with_empty_rank(empty_second_rank):
    # Local TCP avoids FileStore locks/rendezvous files on the shared workspace.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = torch.multiprocessing.spawn(
        _distributed_dice_worker,
        args=(f"tcp://127.0.0.1:{port}", empty_second_rank),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("Distributed Dice workers did not exit within 120 seconds.")
    finally:
        # A process-group timeout cannot interrupt a native destructor deadlock.
        # Bound teardown as well and clean up only this test's own workers.
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
