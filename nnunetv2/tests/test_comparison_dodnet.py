from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from nnunetv2.training.nnUNetTrainer.variants.comparison_baselines.dodnet import (
    DEFAULT_DODNET_TASK_LABELS,
    DoDNetDynamicHead,
    DoDNetUNet,
    merge_task_logits_to_full_scores,
)


@pytest.fixture(autouse=True)
def _single_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _head(out_channels=3):
    return DoDNetDynamicHead(4, 8, 17, out_channels)


def _backbone(deep_supervision=True):
    from dynamic_network_architectures.architectures.unet import PlainConvUNet

    return PlainConvUNet(
        input_channels=1, n_stages=3, features_per_stage=(4, 8, 16),
        conv_op=nn.Conv3d, kernel_sizes=[3] * 3, strides=(1, 2, 2),
        n_conv_per_stage=(1, 1, 1), num_classes=32,
        n_conv_per_stage_decoder=(1, 1), conv_bias=True,
        nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": False},
        deep_supervision=deep_supervision,
    )


@pytest.mark.parametrize("out_channels,expected", [(1, 153), (2, 162), (3, 171)])
def test_parameter_count_matches_original_head(out_channels, expected):
    head = _head(out_channels)
    assert head.num_dynamic_params == expected
    assert head.controller.out_channels == expected
    assert head.controller.in_channels == 8 + 17


def _reference_dynamic_convs(features, parameters, out_channels):
    # Independent per-example implementation. This verifies batch-group kernel
    # layout and every weight/bias boundary, not just matching output shape.
    outputs = []
    for sample in range(features.shape[0]):
        p = parameters[sample]
        weight1 = p[:64].view(8, 8, 1, 1, 1)
        weight2 = p[64:128].view(8, 8, 1, 1, 1)
        weight3 = p[128:128 + 8 * out_channels].view(out_channels, 8, 1, 1, 1)
        bias_offset = 128 + 8 * out_channels
        bias1 = p[bias_offset:bias_offset + 8]
        bias2 = p[bias_offset + 8:bias_offset + 16]
        bias3 = p[bias_offset + 16:]
        value = F.relu(F.conv3d(features[sample:sample + 1], weight1, bias1))
        value = F.relu(F.conv3d(value, weight2, bias2))
        outputs.append(F.conv3d(value, weight3, bias3))
    return torch.cat(outputs)


@pytest.mark.parametrize("out_channels", [1, 2, 3])
def test_grouped_convolution_matches_individual_examples_and_gradients(out_channels):
    torch.manual_seed(12)
    head = _head(out_channels)
    # Transpose makes the input non-contiguous; reshape must preserve samples.
    features = torch.randn(3, 8, 2, 3, 4, dtype=torch.float64).transpose(-1, -2).requires_grad_()
    parameters = torch.randn(3, head.num_dynamic_params, dtype=torch.float64, requires_grad=True)
    actual = head.apply_dynamic_parameters(features, parameters)
    expected = _reference_dynamic_convs(features, parameters, out_channels)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual_grad = torch.autograd.grad(actual.square().mean(), (features, parameters), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().mean(), (features, parameters))
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-11)


def test_controller_uses_both_image_context_and_one_hot_task():
    head = _head()
    with torch.no_grad():
        head.controller.weight.zero_()
        head.controller.bias.zero_()
        head.controller.weight[0, 0, 0, 0, 0] = 2
        head.controller.weight[0, 8 + 2, 0, 0, 0] = 3
    context = torch.zeros(3, 8, 1, 1, 1)
    context[1, 0] = 1
    parameters = head.generate_parameters(context, torch.tensor([0, 0, 2]))
    torch.testing.assert_close(parameters[:, 0], torch.tensor([0., 2., 3.]))


def test_head_batch_permutation_and_singleton_bottleneck():
    torch.manual_seed(42)
    head = _head()
    features, context = torch.randn(3, 4, 2, 2, 2), torch.randn(3, 8, 1, 1, 1)
    ids = torch.tensor([3, 1, 6])
    output = head(features, context, ids)
    permutation = torch.tensor([2, 0, 1])
    torch.testing.assert_close(head(features[permutation], context[permutation], ids[permutation]),
                               output[permutation])
    torch.testing.assert_close(head(features[:1], context[:1], ids[:1]), output[:1])


@pytest.mark.parametrize("ids", [torch.tensor([-1]), torch.tensor([17]), torch.tensor([1.]),
                                  torch.tensor([True]), torch.tensor([[1]]), torch.tensor([1, 2])])
def test_invalid_task_ids_fail_explicitly(ids):
    with pytest.raises(ValueError, match="task_ids"):
        _head().generate_parameters(torch.zeros(1, 8, 1, 1, 1), ids)


def test_default_tasks_cover_cellmap_once_without_special_anchor_path():
    assert len(DEFAULT_DODNET_TASK_LABELS) == 17
    assert DEFAULT_DODNET_TASK_LABELS[:2] == ((4, 5, 6), (7, 8))
    assert DEFAULT_DODNET_TASK_LABELS[12:] == ((17, 18), (1,), (2,), (3,), (27,))
    assert sorted(label for labels in DEFAULT_DODNET_TASK_LABELS for label in labels) == list(range(1, 32))


def test_full_scores_follow_original_order_and_exclude_inactive_slots():
    task_logits = []
    for labels in DEFAULT_DODNET_TASK_LABELS:
        value = torch.full((1, 3, 1, 1, 1), 99999.)
        for slot, label in enumerate(labels):
            value[:, slot] = label
        task_logits.append(value.requires_grad_())
    scores = merge_task_logits_to_full_scores(task_logits)
    torch.testing.assert_close(scores.flatten(), torch.arange(32, dtype=torch.float32))
    scores.sum().backward()
    for logits, labels in zip(task_logits, DEFAULT_DODNET_TASK_LABELS):
        assert torch.all(logits.grad[:, :len(labels)] == 1)
        assert torch.all(logits.grad[:, len(labels):] == 0)


def test_background_threshold_and_foreground_competition_are_explicit():
    logits = [torch.tensor([[[[[-1., 0., 0.1, 2.]]], [[[3., -2., -1., 1.]]]]])]
    scores = merge_task_logits_to_full_scores(logits, ((1, 2),), 3)
    assert scores.argmax(1).flatten().tolist() == [2, 0, 1, 1]
    assert torch.all(scores[:, 0] == 0)


@pytest.mark.parametrize("labels", [((1,), (1,)), ((0,), (1,)), ((1,), (3,)), ((), (1, 2))])
def test_invalid_task_mapping_is_rejected(labels):
    logits = [torch.zeros(1, 3, 1, 1, 1) for _ in labels]
    with pytest.raises(ValueError):
        merge_task_logits_to_full_scores(logits, labels, 3)


def test_plainconv_wrapper_deep_supervision_all_parameters_participate():
    torch.manual_seed(10)
    model = DoDNetUNet(_backbone(True))
    outputs = model(torch.randn(2, 1, 8, 8, 8), torch.tensor([0, 13]))
    assert [o.shape for o in outputs] == [(2, 3, 8, 8, 8), (2, 3, 4, 4, 4)]
    sum(F.binary_cross_entropy_with_logits(o, torch.zeros_like(o)) for o in outputs).backward()
    missing_grad = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    assert missing_grad == []
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(isinstance(head, DoDNetDynamicHead) for head in model.decoder.seg_layers)
    assert not any("film" in name or "anchor" in name for name, _ in model.named_parameters())


@pytest.mark.parametrize("deep_supervision", [False, True])
def test_shared_cached_features_equal_forward_and_all_task_mapping(deep_supervision):
    torch.manual_seed(11)
    model = DoDNetUNet(_backbone(deep_supervision)).eval()
    image = torch.randn(1, 1, 8, 8, 8)
    task = torch.tensor([0])
    with torch.no_grad():
        cache = model.extract_features(image)
        cached = model.predict_from_features(cache, task)
        direct = model(image, task)
        full = model.predict_from_features(cache)
    cached_list = cached if deep_supervision else [cached]
    direct_list = direct if deep_supervision else [direct]
    full_list = full if deep_supervision else [full]
    for a, b, all_tasks in zip(cached_list, direct_list, full_list):
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(all_tasks[:, (4, 5, 6)], a)
        assert all_tasks.shape[1] == 32
        assert torch.all(all_tasks[:, 0] == 0)
    assert all(feature.shape[1] == 8 for feature in cache.projected)


def test_all_task_forward_runs_shared_backbone_once_and_supports_ds_toggle():
    model = DoDNetUNet(_backbone(True)).eval()
    calls = []
    hooks = [module.register_forward_hook(lambda *args: calls.append(1))
             for module in [model.encoder, *model.decoder.stages]]
    model.decoder.deep_supervision = False
    try:
        with torch.no_grad():
            output = model(torch.randn(1, 1, 8, 8, 8))
    finally:
        for hook in hooks:
            hook.remove()
    assert output.shape == (1, 32, 8, 8, 8)
    assert len(calls) == 1 + len(model.decoder.stages)


def test_state_dict_round_trip_and_cpu_autocast_backward():
    model = DoDNetUNet(_backbone(True))
    restored = DoDNetUNet(_backbone(True))
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    image, tasks = torch.randn(1, 1, 8, 8, 8), torch.tensor([16])
    with torch.no_grad():
        for left, right in zip(model(image, tasks), restored(image, tasks)):
            torch.testing.assert_close(left, right)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        outputs = restored(image, tasks)
        loss = sum(F.binary_cross_entropy_with_logits(o.float(), torch.zeros_like(o).float()) for o in outputs)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in restored.parameters())


def test_custom_task_groups_are_supported_without_cellmap_assumptions():
    model = DoDNetUNet(_backbone(False), num_tasks=2, out_channels=2,
                       task_labels=((2, 3), (1,)))
    with torch.no_grad():
        assert model(torch.randn(1, 1, 8, 8, 8)).shape == (1, 4, 8, 8, 8)


def _ddp_worker(rank, rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        model = DistributedDataParallel(DoDNetUNet(_backbone(True)), find_unused_parameters=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        # Different tasks across ranks, including a singleton. The second step
        # catches DDP reducer failures that a single backward does not expose.
        for _ in range(2):
            outputs = model(torch.randn(1, 1, 8, 8, 8), torch.tensor([0 if rank == 0 else 16]))
            active = torch.tensor([1., 1., 1.] if rank == 0 else [1., 0., 0.]).view(1, 3, 1, 1, 1)
            losses = [(F.binary_cross_entropy_with_logits(o, torch.zeros_like(o), reduction="none") * active).mean()
                      for o in outputs]
            # The zero-weight auxiliary scale must remain connected to DDP.
            loss = losses[0] + 0.0 * losses[1]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
                    reason="CPU Gloo distributed backend unavailable")
def test_two_rank_ddp_handles_different_tasks_and_graph_connected_inactive_slots(tmp_path):
    torch.multiprocessing.spawn(_ddp_worker, args=(str(tmp_path / "gloo_init"),), nprocs=2, join=True)
