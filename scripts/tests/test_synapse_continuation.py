"""CPU checks for the explicit global-epoch continuation schedule and guards."""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


path = Path(__file__).resolve()
runner = path.parent / "runner.py"
if not runner.exists():
    runner = path.parents[1] / "continue_synapse_masked.py"
spec = importlib.util.spec_from_file_location("synapse_continuation", runner)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def scheduler(**kwargs):
    optimizer = SimpleNamespace(param_groups=[{"lr": .001}, {"lr": .001}])
    args = dict(start_epoch=1000, end_epoch=2000, start_lr=2e-5, end_lr=1e-6)
    args.update(kwargs)
    return module.TailLR(optimizer, **args)


def test_endpoints_and_optimizer_groups():
    s = scheduler()
    assert s.step(1000) == pytest.approx(2e-5)
    assert all(g["lr"] == pytest.approx(2e-5) for g in s.optimizer.param_groups)
    assert s.step(2000) == pytest.approx(1e-6)
    assert s.step(2001) == pytest.approx(1e-6)


def test_every_epoch_finite_positive_nonincreasing():
    values = [scheduler().value(i) for i in range(1000, 2001)]
    assert all(math.isfinite(v) and 1e-6 <= v <= 2e-5 for v in values)
    assert all(a >= b for a, b in zip(values, values[1:]))


@pytest.mark.parametrize("epoch", [1000, 1025, 1250, 1500, 1750, 1999, 2000])
def test_resume_uses_global_epoch_without_reset(epoch):
    uninterrupted = scheduler()
    for e in range(1000, epoch + 1):
        uninterrupted.step(e)
    resumed = scheduler()
    resumed.step(epoch)
    assert resumed.optimizer.param_groups == uninterrupted.optimizer.param_groups


@pytest.mark.parametrize("kwargs", [dict(end_epoch=1000), dict(start_epoch=-1), dict(start_lr=0),
                                   dict(end_lr=0), dict(end_lr=3e-5), dict(start_lr=float("nan")),
                                   dict(exponent=0), dict(end_lr=float("inf"))])
def test_reject_invalid_schedule(kwargs):
    with pytest.raises(ValueError):
        scheduler(**kwargs)


@pytest.mark.parametrize("epoch", [999, float("nan"), float("inf")])
def test_reject_invalid_epoch(epoch):
    with pytest.raises(ValueError):
        scheduler().step(epoch)


def test_modified_locked_file_rejected(tmp_path):
    path = tmp_path / "input"
    path.write_text("original")
    expected = module.digest(path)
    path.write_text("changed")
    with pytest.raises(ValueError, match="Locked input changed"):
        module.verify({"locked_files": {str(path): expected}})


def test_existing_run_rejected_before_loading_parent(tmp_path):
    with pytest.raises(FileExistsError):
        module.prepare(SimpleNamespace(run_root=tmp_path, parent_run=tmp_path / "missing"))


def test_state_comparison_handles_nan_and_numpy_arrays():
    import numpy as np
    import torch
    a = {"logging": [float("nan"), np.array([1., np.nan])], "tensor": torch.tensor([2.])}
    b = {"logging": [float("nan"), np.array([1., np.nan])], "tensor": torch.tensor([2.])}
    module.assert_state_equal(a, b)
    b["tensor"] += 1
    with pytest.raises(AssertionError):
        module.assert_state_equal(a, b)


def test_optimizer_momentum_resume_and_global_tail_are_equivalent():
    import copy
    import torch
    weight = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([weight], lr=.001, momentum=.99, nesterov=True)
    tail = module.TailLR(optimizer, 1000, 2000, 2e-5, 1e-6)
    for epoch in range(1000, 1025):
        tail.step(epoch)
        weight.grad = weight.detach().clone()
        optimizer.step()
    restored_weight = torch.nn.Parameter(weight.detach().clone())
    restored_optimizer = torch.optim.SGD([restored_weight], lr=.01, momentum=.99, nesterov=True)
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    module.assert_state_equal(restored_optimizer.state_dict(), optimizer.state_dict())
    restored_tail = module.TailLR(restored_optimizer, 1000, 2000, 2e-5, 1e-6)
    for epoch in range(1025, 1050):
        for w, opt, sched in ((weight, optimizer, tail), (restored_weight, restored_optimizer, restored_tail)):
            sched.step(epoch)
            w.grad = w.detach().clone()
            opt.step()
    torch.testing.assert_close(weight, restored_weight, rtol=0, atol=0)
    module.assert_state_equal(restored_optimizer.state_dict(), optimizer.state_dict())


@pytest.mark.parametrize("fold", [0, 1, 2, 6])
def test_model_path_uses_requested_fold(fold):
    assert module.model_subdir(fold).endswith(f"/fold_{fold}")


@pytest.mark.parametrize("fold", [-1, "0", 0.5, True])
def test_model_path_rejects_invalid_fold(fold):
    with pytest.raises(ValueError, match="nonnegative integer"):
        module.model_subdir(fold)


@pytest.fixture
def parent_factory(tmp_path, monkeypatch):
    """Tiny on-disk inputs exercise preparation without loading a GPU model."""
    def make(fold):
        parent, data = tmp_path / "parent", tmp_path / "data"
        source = parent / "source"
        evaluator = source / "scripts/evaluate_synapse_consistency.py"
        evaluator.parent.mkdir(parents=True)
        evaluator.write_text("# immutable evaluator fixture\n")
        module.save(parent / "source_sha256.json", {
            "scripts/evaluate_synapse_consistency.py": module.digest(evaluator)})
        pp = data / "nnUNet_preprocessed/Dataset210_Synapse13"
        pp.mkdir(parents=True)
        splits = [{"train": ["train0", "train_extra"], "val": ["validation"]},
                  {"train": ["train0"], "val": ["validation"]}]
        module.save(pp / "splits_final.json", splits)
        module.save(pp / "nnUNetPlans.json", {})
        files = pp / "nnUNetPlans_3d_fullres"
        files.mkdir()
        for case in splits[fold]["train"] + splits[fold]["val"]:
            for suffix in (".npy", "_seg.npy", ".pkl"):
                (files / (case + suffix)).write_bytes(b"fixture")
        raw = data / "nnUNet_raw/Dataset210_Synapse13"
        raw.mkdir(parents=True)
        module.save(raw / "dataset.json", {})
        model = parent / "models" / module.model_subdir(fold)
        model.mkdir(parents=True)
        reference = {"arm": "masked", "fold": fold, "epochs": 1000, "trainer": module.TRAINER,
                     "code_root": str(source), "iterations_per_epoch": 250,
                     "train_cases": splits[fold]["train"], "validation_cases": splits[fold]["val"],
                     "splits_sha256": module.digest(pp / "splits_final.json"),
                     "plans_sha256": module.digest(pp / "nnUNetPlans.json")}
        module.save(model / "experiment_contract.json", reference)
        for name, epoch in (("final", 1000), ("best", 963)):
            checkpoint = model / f"checkpoint_{name}.pth"
            checkpoint.write_bytes(f"fold-{fold}-{name}".encode())
            output = parent / ("evaluation" if fold == 1 else f"evaluation_fold{fold}") / "masked" / name
            (output / "masked").mkdir(parents=True)
            module.save(output / "summary.json", {"manifest": {
                "fold": fold, "mode": "masked", "tta": False, "native_export": True,
                "epoch": epoch, "checkpoint_sha256": module.digest(checkpoint),
                "trainer_name": module.TRAINER, "cases": reference["validation_cases"],
                "splits_sha256": reference["splits_sha256"], "plans_sha256": reference["plans_sha256"],
                "evaluator_sha256": module.digest(evaluator)}})
            module.save(output / "masked/summary_native.json", {})
        # Also works when this test suite is copied into a frozen run directory.
        fake_repo = tmp_path / "repo"
        for relative in ("scripts/continue_synapse_masked.py", "scripts/slurm/synapse_masked_continuation.sbatch",
                         "scripts/tests/test_synapse_continuation.py"):
            target = fake_repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# fixture\n")
        monkeypatch.setattr(module, "__file__", str(fake_repo / "scripts/continue_synapse_masked.py"))
        return SimpleNamespace(run_root=tmp_path / "tail", parent_run=parent, data_root=data,
                               fold=fold, epochs=2000, tail_lr=2e-5, end_lr=1e-6, submit=False)
    return make


@pytest.mark.parametrize("fold", [0, 1])
def test_prepare_locks_correct_fold_and_smoke_gates_jobs(parent_factory, monkeypatch, fold):
    args = parent_factory(fold)
    calls = []
    def submit(command, text, env):
        calls.append((command, env))
        return str(7000 + len(calls)) + "\n"
    monkeypatch.setattr(module.subprocess, "check_output", submit)
    args.submit = True
    module.prepare(args)
    manifest = module.read(args.run_root / "submission.json")
    assert manifest["fold"] == manifest["reference_contract"]["fold"] == fold
    assert manifest["additional_epochs"] == 1000
    assert manifest["additional_updates"] == 250000
    assert manifest["parent_checkpoints"]["final"].endswith(f"fold_{fold}/checkpoint_final.pth")
    assert manifest["reference_contract"]["train_cases"] == (["train0", "train_extra"] if fold == 0 else ["train0"])
    assert len(calls) == 3
    for (command, env), phase in zip(calls, ("smoke", "train", "evaluate")):
        assert f"--job-name=as_mask_f{fold}_2000ep_{phase}" in command
        assert env["PHASE"] == phase and env["RUN_ROOT"] == str(args.run_root)
    assert "--dependency=afterok:7001" in calls[1][0]
    assert "--dependency=afterok:7002" in calls[2][0]
    module.verify(manifest)


@pytest.mark.parametrize("field,value", [("fold", 1), ("tta", True), ("native_export", False),
                                         ("epoch", 999), ("trainer_name", "wrong"),
                                         ("cases", ["wrong"]), ("splits_sha256", "wrong"),
                                         ("evaluator_sha256", "wrong"), ("checkpoint_sha256", "wrong")])
def test_prepare_rejects_mismatched_parent_evaluation(parent_factory, field, value):
    args = parent_factory(0)
    report_path = args.parent_run / "evaluation_fold0/masked/final/summary.json"
    report = module.read(report_path)
    report["manifest"][field] = value
    module.save(report_path, report)
    with pytest.raises(ValueError):
        module.prepare(args)
    assert not args.run_root.exists()


@pytest.mark.parametrize("fold", [0, 1])
def test_evaluate_passes_manifest_fold_and_uses_its_checkpoints(tmp_path, monkeypatch, fold):
    keys = ("cases", "organs", "fold", "patch", "step_fraction", "tta", "stitching",
            "native_export", "splits_sha256", "plans_sha256", "evaluator_sha256")
    protocol = {key: "fixture" for key in keys}
    protocol.update(fold=fold, tta=False, native_export=True)
    manifest = {"run_root": str(tmp_path), "source_root": str(tmp_path / "source"), "data_root": "data",
                "reference_contract": {"fold": fold}, "schedule": {"end_epoch": 2000},
                "parent_evaluations": {}, "caveat": "test"}
    folder = tmp_path / "models" / module.model_subdir(fold)
    folder.mkdir(parents=True)
    for name in ("final", "best"):
        (folder / f"checkpoint_{name}.pth").write_bytes(name.encode())
        path = tmp_path / f"parent_{name}.json"
        module.save(path, {"manifest": {**protocol, "epoch": 1000},
                           "metrics": {"masked": {"native_mean_dice": .8}}})
        manifest["parent_evaluations"][name] = str(path)
    calls = []
    def evaluate(command, check):
        assert check
        assert command[command.index("--fold") + 1] == str(fold)
        assert "--native" in command and "--tta" not in command
        checkpoint = Path(command[command.index("--checkpoint") + 1])
        assert checkpoint.parent == folder
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True)
        epoch = 2000 if output.name == "final" else 1700
        module.save(output / "summary.json", {
            "manifest": {**protocol, "epoch": epoch, "checkpoint_sha256": module.digest(checkpoint)},
            "metrics": {"masked": {"native_mean_dice": .81}}})
        calls.append(command)
    monkeypatch.setattr(module.subprocess, "run", evaluate)
    module.evaluate(manifest)
    comparison = module.read(tmp_path / "comparison_no_tta.json")
    assert comparison["complete"] and len(calls) == 2
    assert all(row["delta_native_dice"] == pytest.approx(.01) for row in comparison["rows"])
