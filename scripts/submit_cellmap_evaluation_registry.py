#!/usr/bin/env python3
"""Submit enabled, unfinished rows from a tab-separated evaluation registry."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nnunetv2.evaluation.cellmap_pipeline import METRICS_NAME, read_registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--script", default="scripts/slurm/evaluate_cellmap_prediction.sbatch")
    parser.add_argument("--dependency")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    submitted = []
    scheduled_outputs = set()
    for row in read_registry(args.registry):
        if row["enabled"].strip().lower() not in {"1", "true", "yes"}:
            continue
        metrics = Path(row["output_dir"]) / METRICS_NAME
        output_key = str(Path(row["output_dir"]).resolve())
        if metrics.is_file():
            submitted.append({"model": row["model"], "resolution": row["resolution"], "status": "already_complete"})
            continue
        if output_key in scheduled_outputs:
            submitted.append({"model": row["model"], "resolution": row["resolution"], "status": "duplicate_output_already_scheduled"})
            continue
        if not Path(row["predictions"]).is_dir():
            submitted.append({"model": row["model"], "resolution": row["resolution"], "status": "waiting_for_predictions"})
            continue
        scheduled_outputs.add(output_key)
        exported = {
            "PREDICTIONS": row["predictions"],
            "TRUTH_LOCK": row["truth_lock"],
            "OUTPUT_DIR": row["output_dir"],
            "MODEL": row["model"],
            "RESOLUTION": row["resolution"],
        }
        command = ["sbatch", "--parsable"]
        if args.dependency:
            command.extend(["--dependency", args.dependency])
        command.extend(["--export", "ALL," + ",".join(f"{key}={value}" for key, value in exported.items()), args.script])
        if args.dry_run:
            submitted.append({**row, "status": "dry_run", "command": command})
            continue
        result = subprocess.run(command, check=True, capture_output=True, text=True, env=os.environ.copy())
        submitted.append({"model": row["model"], "resolution": row["resolution"], "status": "submitted", "job_id": result.stdout.strip().split(";")[0]})
    print(json.dumps({"registry": str(Path(args.registry).resolve()), "jobs": submitted}, indent=2))


if __name__ == "__main__":
    main()
