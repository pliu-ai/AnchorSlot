import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


path = Path(__file__).resolve().parents[1] / "snapshot_synapse_checkpoints.py"
spec = importlib.util.spec_from_file_location("snapshot_synapse", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_snapshot_is_independent_of_live_file(tmp_path):
    source, destination = tmp_path / "live", tmp_path / "snapshot"
    source.write_bytes(b"original")
    report = module.stable_copy(source, destination)
    assert report["sha256"] == module.digest(destination)
    assert source.stat().st_ino != destination.stat().st_ino
    source.write_bytes(b"updated")
    assert destination.read_bytes() == b"original"


def test_snapshot_refuses_overwrite(tmp_path):
    destination = tmp_path / "existing"
    destination.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        module.stable_copy(tmp_path / "missing", destination)
    assert destination.read_bytes() == b"keep"


def test_copy_retries_if_source_changes(tmp_path):
    source, destination = tmp_path / "live", tmp_path / "snapshot"
    source.write_bytes(b"original")
    copyfile = module.shutil.copyfile
    calls = []
    def changed_copy(src, dst):
        copyfile(src, dst)
        calls.append(1)
        if len(calls) == 1:
            source.write_bytes(b"updated")
    with patch.object(module.shutil, "copyfile", side_effect=changed_copy):
        module.stable_copy(source, destination)
    assert len(calls) == 2 and destination.read_bytes() == b"updated"


def test_copy_fails_if_source_never_stabilizes(tmp_path):
    source, destination = tmp_path / "live", tmp_path / "snapshot"
    source.write_bytes(b"original")
    copyfile = module.shutil.copyfile
    def changed_copy(src, dst):
        copyfile(src, dst)
        source.write_bytes(source.read_bytes() + b"x")
    with patch.object(module.shutil, "copyfile", side_effect=changed_copy):
        with pytest.raises(RuntimeError, match="Source changed"):
            module.stable_copy(source, destination)
    assert not destination.exists() and not destination.with_suffix(".partial").exists()


def test_existing_submission_rejected(tmp_path):
    with pytest.raises(FileExistsError):
        module.prepare(SimpleNamespace(output=tmp_path, base_run=tmp_path / "missing"))


@pytest.mark.parametrize("key", ["fold", "epoch", "checkpoint_sha256", "mode", *module.PROTOCOL_KEYS])
def test_report_mismatch_rejected(key):
    protocol = {k: k for k in module.PROTOCOL_KEYS}
    item = {"fold": 0, "epoch": 850, "checkpoint_sha256": "hash"}
    meta = {**protocol, **item, "mode": "masked"}
    module.validate_report({"manifest": meta}, item, {"protocol": protocol})
    meta[key] = "changed"
    with pytest.raises(ValueError):
        module.validate_report({"manifest": meta}, item, {"protocol": protocol})


def selection(**kwargs):
    args = dict(folds=[0, 1], policies=["best", "latest"], base_run=Path("/base"),
                fold0_run=None, continuation_run=Path("/fold1_tail"))
    args.update(kwargs)
    return SimpleNamespace(**args)


def test_default_model_routing_is_unchanged():
    assert module.requested_runs(selection()) == [(0, Path("/base")), (1, Path("/fold1_tail"))]


def test_fold0_latest_only_uses_continuation_not_parent():
    args = selection(folds=[0], policies=["latest"], fold0_run=Path("/fold0_tail"))
    runs = module.requested_runs(args)
    assert runs == [(0, Path("/fold0_tail"))]
    assert module.evaluation_array(len(runs) * len(args.policies)) == "--array=0"


@pytest.mark.parametrize("kwargs", [dict(folds=[]), dict(folds=[0, 0]), dict(folds=[2]),
                                   dict(policies=[]), dict(policies=["latest", "latest"]),
                                   dict(policies=["unknown"])])
def test_invalid_or_duplicate_selections_are_rejected(kwargs):
    with pytest.raises(ValueError):
        module.requested_runs(selection(**kwargs))


@pytest.mark.parametrize("count,expected", [(1, "--array=0"), (2, "--array=0-1%2"),
                                           (4, "--array=0-3%2")])
def test_array_matches_actual_checkpoint_count(count, expected):
    assert module.evaluation_array(count) == expected


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_empty_or_invalid_array_rejected(count):
    with pytest.raises(ValueError):
        module.evaluation_array(count)
