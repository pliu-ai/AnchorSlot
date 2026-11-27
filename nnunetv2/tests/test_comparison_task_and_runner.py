"""Task-label supervision and non-submitting launch-contract checks."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from nnunetv2.training.nnUNetTrainer.variants.comparison_baselines.task_loss import binary_task_loss, sample_task_ids
from nnunetv2.training.nnUNetTrainer.variants.comparison_baselines.trainers import confusion_counts, supervision_weights
from scripts.run_comparison_baseline import parse_args, recipe_preview


def test_taxonomy_matches_published_groups_without_loading_shell_configuration():
    from nnunetv2.training.nnUNetTrainer.variants.comparison_baselines.taxonomy import DYNAMIC_GROUP_LABELS
    from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.label_mapping_er_dynamic import DEFAULT_DYNAMIC_GROUP_SPECS
    assert DYNAMIC_GROUP_LABELS == tuple(spec.original_labels for spec in DEFAULT_DYNAMIC_GROUP_SPECS)
    script = "from nnunetv2.training.nnUNetTrainer.variants.comparison_baselines.hierarchy import SALT_NUM_LOGITS,HSSN_NUM_LOGITS; assert (SALT_NUM_LOGITS,HSSN_NUM_LOGITS)==(45,50)"
    result = subprocess.run([sys.executable, "-c", script],
                            env={**os.environ, "NNUNET_STRUCTCOND_GROUPING_CONFIG_PATH": "/nonexistent/unrelated.json"},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_inactive_task_slots_do_not_affect_loss_or_gradients():
    x = torch.randn(2, 3, 2, 2, 2, requires_grad=True)
    target = torch.tensor([0, 1, 0, 1, 3, 4, -1, 2]).reshape(1, 1, 2, 2, 2).expand(2, -1, -1, -1, -1)
    task = torch.tensor([13, 0])  # singleton ecs and three-slot mito
    changed = x.detach().clone()
    changed[0, 1:] = 30000
    torch.testing.assert_close(binary_task_loss(x, target, task), binary_task_loss(changed, target, task))
    binary_task_loss(x, target, task).backward()
    assert not x.grad[0, 1:].any()
    assert torch.isfinite(x.grad).all()


def test_all_ignore_task_batch_is_safe_under_half_precision():
    x = torch.full((2, 3, 8, 8, 8), 30000., dtype=torch.float16, requires_grad=True)
    loss = binary_task_loss(x, torch.full((2, 1, 8, 8, 8), -1), torch.tensor([0, 13]))
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(x.grad).all() and not x.grad.any()


@pytest.mark.parametrize("task", [torch.tensor([17]), torch.tensor([-1]), torch.tensor([.5])])
def test_bad_task_id_rejected(task):
    with pytest.raises(ValueError):
        binary_task_loss(torch.zeros(1, 3, 2, 2, 2), torch.zeros(1, 1, 2, 2, 2), task)


def test_task_sampler_covers_anchor_and_dynamic_and_absent_tasks():
    target = torch.ones(4, 1, 2, 2, 2)
    assert (sample_task_ids(target, np.random.default_rng(1), p_present=1) == 13).all()
    absent = sample_task_ids(target, np.random.default_rng(1), p_present=0)
    assert (absent != 13).all()
    assert len(set(sample_task_ids(torch.zeros(128, 1, 1, 1, 1), np.random.default_rng(2)))) == 17


def test_confusion_ignores_invalid_voxels_without_mutating_target():
    scores = torch.zeros(1, 32, 1, 1, 4)
    for position, label in enumerate([1, 2, 2, 3]):
        scores[0, label, 0, 0, position] = 10
    target = torch.tensor([1, 1, -1, 99]).reshape(1, 1, 1, 1, 4)
    saved = target.clone()
    tp, fp, fn = confusion_counts(scores, target, ignore_label=99)
    assert tp[0] == 1 and fp[1] == 1 and fn[0] == 1
    assert tp.sum() == fp.sum() == fn.sum() == 1
    torch.testing.assert_close(target, saved)


def test_deep_supervision_weights_single_level_and_auxiliary_zero():
    np.testing.assert_equal(supervision_weights(1), [1])
    weights = supervision_weights(5)
    assert weights[-1] == 0 and weights.sum() == pytest.approx(1)


def test_runner_is_preview_only_and_checks_split(tmp_path):
    dataset = tmp_path / "Dataset205_test"
    dataset.mkdir()
    (dataset / "nnUNetPlans.json").write_text(json.dumps({"configurations": {"3d_lowres_large_patch": {}}}))
    (dataset / "dataset.json").write_text("{}")
    split_path = dataset / "splits_final.json"
    split_path.write_text(json.dumps([{"train": ["a", "b"], "val": ["c"]}]))
    output = tmp_path / "unused_results"
    args = parse_args(["--method", "salt", "--dataset", dataset.name,
                       "--preprocessed-root", str(tmp_path), "--results-root", str(output)])
    preview = recipe_preview(args)
    assert not args.run and preview["train_cases"] == 2 and preview["val_cases"] == 1
    assert not output.exists()
    assert all(len(value) == 64 for value in preview["metadata_sha256"].values())
    split_path.write_text(json.dumps([{"train": ["a"], "val": ["a"]}]))
    with pytest.raises(ValueError, match="overlaps"):
        recipe_preview(args)


@pytest.mark.parametrize("extra", [["--batch-size", "0"], ["--num-gpus", "5"], ["--device", "cpu"],
                                    ["--dataset", "../Dataset205_test"]])
def test_runner_rejects_invalid_launch(extra):
    with pytest.raises(SystemExit):
        parse_args(["--method", "blob", "--results-root", "/tmp/not_used"] + extra)
