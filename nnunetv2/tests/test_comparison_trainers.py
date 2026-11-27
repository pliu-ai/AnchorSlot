"""CPU integration checks using actual plans, trainers, checkpoints and inference.

All paths are temporary. No dataset is read and no GPU allocation is required.
The distributed test bypasses only the upstream constructor's unconditional CUDA
choice when a process group already exists, not the comparison train step.
"""
from __future__ import annotations

import copy
from datetime import timedelta
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from nnunetv2.training.nnUNetTrainer.variants.comparison_baselines.trainers import (
    TRAINERS,
    confusion_counts,
)
from nnunetv2.training.nnUNetTrainer.variants.comparison_baselines.task_loss import (
    binary_task_loss,
    sample_task_ids,
)


METHODS = tuple(TRAINERS)


def _plans():
    return {
        "dataset_name": "Dataset999_ComparisonCPU", "plans_name": "nnUNetPlans",
        "transpose_forward": [0, 1, 2], "transpose_backward": [0, 1, 2],
        "original_median_spacing_after_transp": [4., 4., 4.],
        "original_median_shape_after_transp": [8, 8, 8],
        "image_reader_writer": "SimpleITKIO", "experiment_planner_used": "ExperimentPlanner",
        "foreground_intensity_properties_per_channel": {},
        "configurations": {"3d_test": {
            "data_identifier": "nnUNetPlans_3d_test", "batch_size": 2,
            "patch_size": [8, 8, 8], "median_image_size_in_voxels": [8, 8, 8],
            "spacing": [4., 4., 4.], "batch_dice": False,
            "normalization_schemes": ["ZScoreNormalization"], "use_mask_for_norm": [False],
            "preprocessor_name": "DefaultPreprocessor",
            "resampling_fn_data": "resample_data_or_seg_to_shape",
            "resampling_fn_data_kwargs": {"is_seg": False, "order": 1, "order_z": 0},
            "resampling_fn_seg": "resample_data_or_seg_to_shape",
            "resampling_fn_seg_kwargs": {"is_seg": True, "order": 0, "order_z": 0},
            "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
            "resampling_fn_probabilities_kwargs": {"is_seg": False, "order": 1, "order_z": 0},
            "architecture": {
                "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
                "arch_kwargs": {
                    "n_stages": 3, "features_per_stage": [8, 16, 32],
                    "conv_op": "torch.nn.Conv3d", "kernel_sizes": [[3, 3, 3]] * 3,
                    "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
                    "n_conv_per_stage": [1, 1, 1], "n_conv_per_stage_decoder": [1, 1],
                    "conv_bias": True, "norm_op": "torch.nn.InstanceNorm3d",
                    "norm_op_kwargs": {"affine": True, "eps": 1e-5},
                    "dropout_op": None, "dropout_op_kwargs": None,
                    "nonlin": "torch.nn.LeakyReLU", "nonlin_kwargs": {"inplace": True},
                },
                "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
            },
        }},
    }


def _dataset(ignore=False):
    labels = {"background": 0, **{f"class_{label}": label for label in range(1, 32)}}
    if ignore:
        labels["ignore"] = 32
    return {"channel_names": {"0": "EM"}, "labels": labels,
            "numTraining": 2, "file_ending": ".nii.gz"}


def _batch(batch_size=2, ignore=False, all_ignored=False):
    target = torch.arange(batch_size * 8 ** 3).reshape(batch_size, 1, 8, 8, 8).remainder(32)
    if ignore:
        target[:, :, :2, :2, :2] = 32
    if all_ignored:
        target.fill_(32)
    return {"data": torch.randn(batch_size, 1, 8, 8, 8),
            "target": [target, F.interpolate(target.float(), size=(4, 4, 4), mode="nearest").long()]}


@pytest.fixture(autouse=True)
def _isolated_cpu_environment(monkeypatch, tmp_path):
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    module = importlib.import_module("nnunetv2.training.nnUNetTrainer.nnUNetTrainer")
    monkeypatch.setattr(module, "nnUNet_results", str(tmp_path / "results"))
    monkeypatch.setattr(module, "nnUNet_preprocessed", str(tmp_path / "preprocessed"))
    monkeypatch.delenv("NNUNET_STRUCTCOND_GROUPING_CONFIG_PATH", raising=False)
    monkeypatch.setenv("NNUNET_COMPARISON_BATCH_SIZE", "2")
    monkeypatch.setenv("NNUNET_COMPARISON_LR", "0.001")
    monkeypatch.setenv("NNUNET_COMPARISON_EPOCHS", "3")
    monkeypatch.setenv("NNUNET_COMPARISON_TRAIN_ITERS", "2")
    monkeypatch.setenv("NNUNET_COMPARISON_VAL_ITERS", "1")
    monkeypatch.setenv("nnUNet_compile", "false")
    yield
    torch.set_num_threads(before)


def _trainer(method, ignore=False):
    # Exercise inherited introspection/my_init_kwargs without mocking __init__.
    return TRAINERS[method](_plans(), "3d_test", 0, _dataset(ignore),
                            unpack_dataset=False, device=torch.device("cpu"))


@pytest.mark.parametrize("method", METHODS)
def test_real_trainer_two_train_steps_validation_and_full_inference(method):
    torch.manual_seed(91)
    trainer = _trainer(method, ignore=True)
    assert trainer.my_init_kwargs["configuration"] == "3d_test"
    assert trainer.my_init_kwargs["device"] == torch.device("cpu")
    assert trainer.batch_size == 2 and trainer.label_manager.num_segmentation_heads == 32
    trainer.initialize()
    before = {key: value.detach().clone() for key, value in trainer.network.named_parameters()}
    batch = _batch(ignore=True)
    original_targets = [t.clone() for t in batch["target"]]
    for epoch in range(2):
        trainer.current_epoch = epoch
        result = trainer.train_step(batch)
        assert np.isfinite(result["loss"])
        missing = [key for key, p in trainer.network.named_parameters() if p.grad is None]
        assert not missing, f"parameters absent from autograd: {missing}"
        assert all(torch.isfinite(p.grad).all() for p in trainer.network.parameters())
    assert any(not torch.equal(before[key], p) for key, p in trainer.network.named_parameters())
    trainer.network.eval()
    result = trainer.validation_step(batch)
    assert np.isfinite(result["loss"])
    for name in ("tp_hard", "fp_hard", "fn_hard"):
        assert result[name].shape == (31,)
        assert np.all(np.isfinite(result[name])) and np.all(result[name] >= 0)
    for original, actual in zip(original_targets, batch["target"]):
        torch.testing.assert_close(original, actual)
    # Standard nnU-Net inference must need neither task IDs nor auxiliary flags.
    trainer.set_deep_supervision_enabled(False)
    with torch.no_grad():
        scores = trainer.network(batch["data"])
    assert scores.shape == (2, 32, 8, 8, 8)
    assert torch.isfinite(scores).all()
    if method == "salt":
        torch.testing.assert_close(scores.exp().sum(1), torch.ones_like(scores[:, 0]))
    if trainer.method == "dodnet":
        assert torch.all(scores[:, 0] == 0)


@pytest.mark.parametrize("method", METHODS)
def test_ignored_only_batch_is_graph_connected_and_metrics_empty(method):
    trainer = _trainer(method, ignore=True)
    trainer.initialize()
    batch = _batch(ignore=True, all_ignored=True)
    result = trainer.train_step(batch)
    assert float(result["loss"]) == pytest.approx(0.)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainer.network.parameters())
    metrics = trainer.validation_step(batch)
    assert float(metrics["loss"]) == pytest.approx(0.)
    assert all(np.all(metrics[name] == 0) for name in ("tp_hard", "fp_hard", "fn_hard"))


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("method", METHODS)
def test_checkpoint_strict_resume_and_standard_predictor_export(method, monkeypatch):
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    torch.manual_seed(72)
    trainer = _trainer(method)
    trainer.initialize()
    trainer.train_step(_batch())
    trainer.current_epoch = 1
    checkpoint_path = Path(trainer.output_folder) / "checkpoint_final.pth"
    trainer.save_checkpoint(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_rng_draw = trainer.task_rng.integers(0, 17, size=20)
    restored = _trainer(method)
    restored.load_checkpoint(checkpoint_path)
    assert restored.current_epoch == 2
    _assert_nested_equal(trainer.network.state_dict(), restored.network.state_dict())
    _assert_nested_equal(trainer.optimizer.state_dict(), restored.optimizer.state_dict())
    np.testing.assert_array_equal(restored.task_rng.integers(0, 17, size=20), expected_rng_draw)
    restored.lr_scheduler.step(restored.current_epoch)
    assert restored.optimizer.param_groups[0]["lr"] == pytest.approx(0.001 * (1 - 2 / 3) ** 0.9)
    assert np.isfinite(restored.train_step(_batch())["loss"])

    bad = copy.deepcopy(checkpoint)
    bad["network_weights"].pop(next(iter(bad["network_weights"])))
    with pytest.raises(RuntimeError, match="Missing key"):
        _trainer(method).load_checkpoint(bad)
    bad = copy.deepcopy(checkpoint)
    bad["comparison_recipe"]["method"] = "incompatible_method"
    with pytest.raises(ValueError, match="recipe"):
        _trainer(method).load_checkpoint(bad)

    root = Path(trainer.output_folder_base)
    (root / "plans.json").write_text(json.dumps(_plans()))
    (root / "dataset.json").write_text(json.dumps(_dataset()))
    # Predictor reconstructs from checkpoint metadata, not launch-shell knobs.
    monkeypatch.setenv("NNUNET_STRUCTCOND_GROUPING_CONFIG_PATH", "/nonexistent/grouping.json")
    monkeypatch.setenv("NNUNET_COMPARISON_BATCH_SIZE", "99")
    predictor = nnUNetPredictor(tile_step_size=1., use_gaussian=False, use_mirroring=False,
                                perform_everything_on_device=False, device=torch.device("cpu"),
                                verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(root), use_folds=(0,), checkpoint_name="checkpoint_final.pth")
    image = torch.randn(1, 8, 8, 8)
    scores = predictor.predict_logits_from_preprocessed_data(image)
    assert scores.shape == (32, 8, 8, 8) and torch.isfinite(scores).all()
    trainer.set_deep_supervision_enabled(False)
    trainer.network.eval()
    with torch.no_grad():
        expected = trainer.network(image[None])[0]
    # nnU-Net's sliding-window accumulation is fp16, including on CPU.
    torch.testing.assert_close(scores.float(), expected.float(), rtol=2e-3, atol=2e-3)
    properties = {"spacing": [4., 4., 4.], "shape_after_cropping_and_before_resampling": [8, 8, 8],
                  "shape_before_cropping": [8, 8, 8], "bbox_used_for_cropping": [[0, 8]] * 3}
    segmentation = convert_predicted_logits_to_segmentation_with_correct_shape(
        scores, predictor.plans_manager, predictor.configuration_manager,
        predictor.label_manager, properties, num_threads_torch=1)
    assert segmentation.shape == (8, 8, 8)
    assert segmentation.dtype == np.uint8
    np.testing.assert_array_equal(segmentation, scores.argmax(0).cpu().numpy())


def test_validation_counts_original_classes_and_preserves_ignore_target():
    target = torch.tensor([[[[[1, 2, 31, 0, 32, -1]]]]])
    predictions = torch.tensor([1, 1, 31, 2, 1, 31])
    scores = torch.full((1, 32, 1, 1, 6), -10.)
    scores[0, predictions, 0, 0, torch.arange(6)] = 10
    original = target.clone()
    tp, fp, fn = confusion_counts(scores, target, ignore_label=32)
    assert tp[0] == 1 and tp[30] == 1 and tp.sum() == 2
    assert fp[0] == 1 and fp[1] == 1 and fp.sum() == 2
    assert fn[1] == 1 and fn.sum() == 1
    torch.testing.assert_close(target, original)


def test_task_loss_and_sampler_respect_singleton_anchor_and_unused_slots():
    target = torch.ones(1, 1, 2, 2, 2, dtype=torch.long)
    # ECS is task 13; inactive binary slots must carry no loss or gradient.
    logits = torch.randn(1, 3, 2, 2, 2, requires_grad=True)
    ids = torch.tensor([13])
    expected = binary_task_loss(logits, target, ids)
    perturbed = logits.detach().clone()
    perturbed[:, 1:] = 1000
    torch.testing.assert_close(binary_task_loss(perturbed, target, ids), expected)
    expected.backward()
    assert torch.all(logits.grad[:, 1:] == 0)
    rng = np.random.default_rng(1)
    assert np.all(sample_task_ids(target.repeat(8, 1, 1, 1, 1), rng, p_present=1) == 13)
    assert np.all(sample_task_ids(target.repeat(8, 1, 1, 1, 1), rng, p_present=0) != 13)


@pytest.mark.parametrize("method", METHODS)
def test_environment_batch_override_reaches_actual_training_batch(method, monkeypatch):
    monkeypatch.setenv("NNUNET_COMPARISON_BATCH_SIZE", "4")
    trainer = _trainer(method)
    # The plans stay unchanged for checkpoint reconstruction, while the actual
    # loader/training batch uses the explicit benchmark override.
    assert trainer.configuration_manager.batch_size == 2
    assert trainer.plans_manager.plans["configurations"]["3d_test"]["batch_size"] == 2
    assert trainer.batch_size == 4
    assert trainer.comparison_recipe()["global_batch_size"] == 4
    trainer.initialize()
    observed_batches = []
    hook = trainer.network.register_forward_pre_hook(lambda module, args: observed_batches.append(args[0].shape[0]))
    try:
        result = trainer.train_step(_batch(batch_size=trainer.batch_size))
    finally:
        hook.remove()
    assert observed_batches == [4]
    assert np.isfinite(result["loss"])


class _NonfiniteAuxiliaryOutput(torch.nn.Module):
    """Isolate loss assembly from convolution backward on nonfinite operands."""

    def __init__(self, channels, features=False):
        super().__init__()
        self.main = torch.nn.Parameter(torch.randn(2, channels, 8, 8, 8))
        self.coarse = torch.nn.Parameter(torch.full((2, channels, 4, 4, 4), float("inf")))
        with torch.no_grad():
            self.coarse.reshape(-1)[0] = float("nan")
            self.coarse.reshape(-1)[1] = float("-inf")
        self.features = torch.nn.Parameter(torch.randn(2, 16, 4, 4, 4)) if features else None

    def forward(self, data, task_ids=None, return_aux=False):
        output = [self.main, self.coarse]
        return {"raw": output, "features": self.features} if return_aux else output


@pytest.mark.parametrize("method", METHODS)
def test_zero_weight_auxiliary_nan_inf_produce_finite_loss_and_zero_gradients(method):
    trainer = _trainer(method)
    trainer.loss = trainer._build_loss()
    channels = {"salt": 45, "hssn": 50, "dodnet": 3}.get(method, 32)
    trainer.network = _NonfiniteAuxiliaryOutput(channels, features=method == "hssn")
    trainer.current_epoch = 1
    batch = _batch()
    ids = torch.tensor([0, 13]) if method == "dodnet" else None
    torch.manual_seed(54)
    actual = trainer._forward_loss(batch["data"], batch["target"], ids=ids)
    assert torch.isfinite(actual)
    actual.backward()
    assert trainer.network.coarse.grad is not None
    assert torch.all(trainer.network.coarse.grad == 0)
    assert torch.isfinite(trainer.network.main.grad).all()
    assert torch.any(trainer.network.main.grad != 0)
    # Exact same main objective when the deliberately unused output is finite.
    with torch.no_grad():
        trainer.network.coarse.zero_()
    torch.manual_seed(54)
    reference = trainer._forward_loss(batch["data"], batch["target"], ids=ids)
    torch.testing.assert_close(actual.detach(), reference.detach())


def test_all_task_control_matches_explicit_mean_and_supervises_every_task():
    from nnunetv2.training.nnUNetTrainer.variants.comparison_baselines.dodnet import DEFAULT_DODNET_TASK_LABELS

    trainer = _trainer("dodnet_all")
    trainer.initialize()
    batch = _batch()
    original_rng = copy.deepcopy(trainer.task_rng.bit_generator.state)
    data, targets, ids, components = trainer._prepare(batch)
    assert ids is None and components is None
    assert trainer.task_rng.bit_generator.state == original_rng
    assert trainer.comparison_recipe()["task_supervision"] == "all"
    cache = trainer.network.extract_features(data)
    explicit = []
    for task in range(len(DEFAULT_DODNET_TASK_LABELS)):
        task_ids = torch.full((2,), task, dtype=torch.long)
        local_logits = trainer.network.predict_from_features(cache, task_ids)[0]
        explicit.append(binary_task_loss(local_logits, targets[0], task_ids))
    expected = torch.stack(explicit).mean()
    actual = trainer._forward_loss(data, targets)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    # Every one-hot input column reaches the same final controller on this step.
    head = trainer.network.heads[-1]
    task_column_gradients = head.controller.weight.grad[:, -17:]
    assert torch.all(task_column_gradients.flatten(2).abs().sum((0, 2)) > 0)


def test_sampled_and_all_task_checkpoints_are_not_interchangeable():
    sampled = _trainer("dodnet")
    sampled.initialize()
    checkpoint_path = Path(sampled.output_folder) / "checkpoint_latest.pth"
    sampled.save_checkpoint(str(checkpoint_path))
    with pytest.raises(ValueError, match="recipe"):
        _trainer("dodnet_all").load_checkpoint(checkpoint_path)


def test_all_tasks_supervises_multiple_decoder_scales_with_exact_weights():
    from nnunetv2.training.nnUNetTrainer.variants.comparison_baselines.dodnet import DEFAULT_DODNET_TASK_LABELS
    plans = _plans()
    cfg = plans["configurations"]["3d_test"]
    cfg["patch_size"] = [16, 16, 16]
    cfg["architecture"]["arch_kwargs"].update(
        n_stages=4, features_per_stage=[8, 16, 32, 64], kernel_sizes=[[3, 3, 3]] * 4,
        strides=[[1, 1, 1]] + [[2, 2, 2]] * 3, n_conv_per_stage=[1] * 4,
        n_conv_per_stage_decoder=[1] * 3)
    trainer = TRAINERS["dodnet_all"](plans, "3d_test", 0, _dataset(), device=torch.device("cpu"))
    trainer.initialize()
    data = torch.randn(2, 1, 16, 16, 16)
    target = torch.randint(0, 32, (2, 1, 16, 16, 16))
    targets = [target] + [F.interpolate(target.float(), size=(size,) * 3, mode="nearest").long()
                          for size in (8, 4)]
    cache = trainer.network.extract_features(data)
    expected = 0
    for task in range(len(DEFAULT_DODNET_TASK_LABELS)):
        ids = torch.full((2,), task, dtype=torch.long)
        outputs = trainer.network.predict_from_features(cache, ids)
        expected = expected + (2 / 3 * binary_task_loss(outputs[0], targets[0], ids)
                               + 1 / 3 * binary_task_loss(outputs[1], targets[1], ids)) / 17
    actual = trainer._forward_loss(data, targets)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    for head in trainer.network.heads[-2:]:
        gradient = head.controller.weight.grad[:, -17:]
        assert torch.all(gradient.flatten(2).abs().sum((0, 2)) > 0)
    assert not trainer.network.heads[0].controller.weight.grad.any()


def _distributed_trainers_worker(rank, rendezvous, temporary_root):
    import torch.distributed as dist

    torch.set_num_threads(1)
    base = importlib.import_module("nnunetv2.training.nnUNetTrainer.nnUNetTrainer")
    base.nnUNet_results = str(Path(temporary_root) / f"rank_{rank}" / "results")
    base.nnUNet_preprocessed = str(Path(temporary_root) / "preprocessed")
    os.environ.pop("NNUNET_STRUCTCOND_GROUPING_CONFIG_PATH", None)
    # nnUNetTrainer's constructor assumes initialized DDP implies CUDA. Create
    # before Gloo, then exercise the real comparison initialize/train methods.
    trainers = [_trainer(method, ignore=True) for method in METHODS]
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        for trainer in trainers:
            trainer.is_ddp, trainer.local_rank = True, rank
            trainer.task_rng = np.random.default_rng(1234 + rank)
            trainer._set_batch_size_and_oversample()
            assert trainer.batch_size == 1
            trainer.initialize()
            for epoch in range(2):
                trainer.current_epoch = epoch
                result = trainer.train_step(_batch(1, ignore=True))
                assert np.isfinite(result["loss"])
                assert all(p.grad is not None and torch.isfinite(p.grad).all()
                           for p in trainer.network.parameters())
            trainer.network.eval()
            metrics = trainer.validation_step(_batch(1, ignore=True))
            assert metrics["tp_hard"].shape == (31,)
            checksum = torch.stack([p.detach().double().sum() for p in trainer.network.parameters()]).sum()
            checksums = [torch.empty_like(checksum) for _ in range(2)]
            dist.all_gather(checksums, checksum)
            torch.testing.assert_close(checksums[0], checksums[1], rtol=0, atol=0)
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
                    reason="Gloo unavailable")
def test_two_rank_ddp_all_comparison_trainers(tmp_path):
    torch.multiprocessing.spawn(_distributed_trainers_worker,
                                args=(str(tmp_path / "gloo_init"), str(tmp_path)), nprocs=2, join=True)
