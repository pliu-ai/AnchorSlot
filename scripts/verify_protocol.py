#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check AnchorSlot paper/reference protocol metadata.")
    parser.add_argument("--config-root", type=Path, default=Path("configs/cellmap"))
    parser.add_argument("--raw-dataset", type=Path, default=None)
    args = parser.parse_args()

    paper = load_json(args.config_root / "paper_dataset200" / "dataset.json")
    reference = load_json(args.config_root / "reference_dataset202" / "dataset.json")
    if int(paper["numTraining"]) <= 0:
        raise RuntimeError("Dataset200 has no training cases")
    reference_description = str(reference.get("description", "")).lower()
    if "imagests" not in reference_description or "labelsts" not in reference_description:
        raise RuntimeError("Dataset202 leakage warning is no longer supported by its metadata")

    print(f"paper protocol: {paper['name']} ({paper['numTraining']} training cases)")
    print("reference protocol: Dataset202 is leakage-positive and historical only")

    if args.raw_dataset is not None:
        def case_ids(folder: Path) -> set[str]:
            return {
                path.name.removesuffix(".nii.gz").rsplit("_", 1)[0]
                for path in folder.glob("*_0000.nii.gz")
            }

        train_ids = case_ids(args.raw_dataset / "imagesTr")
        test_ids = case_ids(args.raw_dataset / "imagesTs")
        overlap = sorted(train_ids & test_ids)
        if overlap:
            raise RuntimeError(f"raw Dataset200 train/test overlap: {overlap}")
        expected_training = int(paper["numTraining"])
        if len(train_ids) != expected_training:
            raise RuntimeError(
                f"paper metadata declares {expected_training} training cases, "
                f"but the raw dataset has {len(train_ids)}"
            )
        print(f"raw split: {len(train_ids)} train, {len(test_ids)} test, zero filename overlap")


if __name__ == "__main__":
    main()
