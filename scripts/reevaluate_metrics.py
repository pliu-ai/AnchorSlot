import argparse
import json
import os
from collections import defaultdict
from glob import glob
from typing import Any, Dict, List

METRIC_KEYS = ["Dice", "FN", "FP", "IoU", "TN", "TP", "n_pred", "n_ref"]


def mean_of(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def filter_summary_by_present_labels(
    summary: Dict[str, Any], unique_labels: Dict[str, List[int]]
) -> Dict[str, Any]:
    cases = summary["metric_per_case"]
    filtered_cases = []
    label_accum: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    fg_metric_totals: Dict[str, List[float]] = defaultdict(list)
    unmatched: List[str] = []

    for case in cases:
        fname = os.path.basename(case["prediction_file"])
        if fname not in unique_labels:
            unmatched.append(fname)
            continue

        present_labels = {str(l) for l in unique_labels[fname]}
        filtered_metrics = {
            lbl: mvals for lbl, mvals in case["metrics"].items() if lbl in present_labels
        }
        filtered_cases.append(
            {
                "metrics": filtered_metrics,
                "prediction_file": case["prediction_file"],
                "reference_file": case["reference_file"],
            }
        )

        for lbl, mvals in filtered_metrics.items():
            for key in METRIC_KEYS:
                if key in mvals:
                    label_accum[lbl][key].append(mvals[key])
            if lbl != "0":
                for key in METRIC_KEYS:
                    if key in mvals:
                        fg_metric_totals[key].append(mvals[key])

    mean_per_label: Dict[str, Dict[str, float]] = {}
    for lbl, accum in sorted(label_accum.items(), key=lambda x: int(x[0])):
        mean_per_label[lbl] = {k: mean_of(v) for k, v in accum.items()}
        mean_per_label[lbl]["n_cases"] = len(accum.get("Dice", []))

    foreground_mean = {k: mean_of(v) for k, v in fg_metric_totals.items()}
    n_fg = len(fg_metric_totals.get("Dice", []))
    foreground_mean["n_fg_label_cases"] = n_fg

    return {
        "foreground_mean": foreground_mean,
        "mean": mean_per_label,
        "metric_per_case": filtered_cases,
        "_meta": {
            "unmatched_filenames": unmatched,
            "n_input_cases": len(cases),
            "n_filtered_cases": len(filtered_cases),
            "n_fg_label_cases": n_fg,
        },
    }


def process_one_summary(
    summary_path: str, unique_labels: Dict[str, List[int]], verbose: bool = True
) -> str:
    summary = load_json(summary_path)
    filtered = filter_summary_by_present_labels(summary, unique_labels)

    output_path = os.path.join(os.path.dirname(summary_path), "summary_filtered.json")
    output_dict = {
        "foreground_mean": filtered["foreground_mean"],
        "mean": filtered["mean"],
        "metric_per_case": filtered["metric_per_case"],
    }
    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=4)

    if verbose:
        unmatched = filtered["_meta"]["unmatched_filenames"]
        if unmatched:
            print(f"WARNING: {len(unmatched)} unmatched in {summary_path}: {unmatched}")

        print(f"[DONE] {summary_path}")
        print(f"       -> {output_path}")
        print(
            "       cases: "
            f"{filtered['_meta']['n_filtered_cases']} / {filtered['_meta']['n_input_cases']}"
        )
        print(
            "       fg Dice/IoU: "
            f"{filtered['foreground_mean'].get('Dice', float('nan')):.4f} / "
            f"{filtered['foreground_mean'].get('IoU', float('nan')):.4f}"
        )
        print()

    return output_path


def collect_summary_paths(input_root: str) -> List[str]:
    # Prioritize grouped predictions (group_00...group_11). If none exists, fall back to root summary.
    grouped = sorted(glob(os.path.join(input_root, "group_*", "summary.json")))
    if grouped:
        return grouped

    root_summary = os.path.join(input_root, "summary.json")
    if os.path.isfile(root_summary):
        return [root_summary]
    return []


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate nnUNet summary metrics by filtering class-wise metrics "
            "with per-case present labels. For grouped predictions, this processes "
            "all group_*/summary.json and writes summary_filtered.json beside each summary."
        )
    )
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Directory containing group_*/summary.json or a root summary.json.",
    )
    parser.add_argument(
        "--unique_labels_path",
        type=str,
        required=True,
        help="Path to labelsTs_unique_labels_sum.json.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    input_root = args.input_root
    unique_labels_path = args.unique_labels_path
    verbose = not args.quiet

    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"input_root not found: {input_root}")
    if not os.path.isfile(unique_labels_path):
        raise FileNotFoundError(f"unique_labels_path not found: {unique_labels_path}")

    summary_paths = collect_summary_paths(input_root)
    if not summary_paths:
        raise RuntimeError(
            f"No summary.json found under {input_root}. "
            "Expected group_*/summary.json or summary.json."
        )

    unique_labels = load_json(unique_labels_path)
    written_outputs: List[str] = []
    for summary_path in summary_paths:
        written_outputs.append(process_one_summary(summary_path, unique_labels, verbose))

    print(f"Processed {len(summary_paths)} summary files.")
    print("Output files:")
    for output_path in written_outputs:
        print(f"  {output_path}")


if __name__ == "__main__":
    main()
