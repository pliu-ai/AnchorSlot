#!/usr/bin/env python3
"""Official 218-crop 5-fold CV verdict for Step-1 upgraded decoders vs Flat nnU-Net and AnchorSlot (Gmix0.5)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import numpy as np

SRC = "/projects/weilab/liupeng/runs/Anchorslot/validation_roi_audit_20261004_ghgmix05/source_dependencies/report_refresh.py"
ns = {}
exec(
    compile(
        open(SRC).read().replace("from refresh_cellmap import ROOT, TASKS, EXISTING, COMMIT, LOCK, read, write, sha", ""),
        "rr",
        "exec",
    ),
    ns,
)
combine, srx = ns["combine"], ns["signed_rank_exact"]

CV5 = Path("/projects/weilab/liupeng/runs/Anchorslot/cv5_ds205_20261008/eval")
V2 = Path("/projects/weilab/liupeng/runs/Anchorslot/upgrade_v2_20261121/step1_cv5_validation/eval")
METRICS = ("overall", "semantic", "instance")
VARIANTS = ("TS_w0.65_g0.15", "TS_w0.65_g0.15_veto")


def load_metrics(path: Path):
    m = json.loads(path.read_text())
    c = {k: v for k, v in m.items() if k.startswith("crop")}
    assert abs(combine(c)["overall"] - m["overall_score"]) < 1e-12
    assert m["num_evals_done"] == m["total_evals"]
    return c


def compare_pair(name_a: str, A: dict, name_b: str, B: dict, folds_a: dict, folds_b: dict):
    ks = sorted(A)
    n = len(ks)
    pa, pb = combine(A), combine(B)
    rng = np.random.default_rng(20261008)
    boot = {m: [] for m in METRICS}
    for _ in range(10000):
        idx = rng.integers(0, n, n)
        sel = [ks[i] for i in idx]
        ab = combine({f"{j}:{c}": A[c] for j, c in enumerate(sel)})
        bb = combine({f"{j}:{c}": B[c] for j, c in enumerate(sel)})
        for m in METRICS:
            boot[m].append(bb[m] - ab[m])
    out = {}
    print(f"\n=== {name_b} vs {name_a} (pooled {n} crops across {len(folds_a)} folds) ===")
    for m in METRICS:
        diff = pb[m] - pa[m]
        fw = sum(folds_b[f][m] > folds_a[f][m] for f in folds_a)
        x = np.array([combine({c: B[c]})[m] for c in ks])
        y = np.array([combine({c: A[c]})[m] for c in ks])
        p = float(srx(x, y))
        lo, hi = [float(v) for v in np.percentile(boot[m], [2.5, 97.5])]
        ok = diff > 0 and fw >= 3 and p < 0.05
        out[m] = {
            "base": pa[m], "new": pb[m], "diff": diff,
            "ci95": [lo, hi], "folds_won": fw,
            "wins": int((x > y).sum()), "ties": int((x == y).sum()),
            "p": p, "met": bool(ok),
        }
        print(
            f"  {m:9s} {name_a}={pa[m]:.4f}  {name_b}={pb[m]:.4f}  Δ={diff:+.4f} "
            f"CI=[{lo:+.4f}, {hi:+.4f}]  folds={fw}/{len(folds_a)}  "
            f"crops={int((x > y).sum())}/{n} (ties {int((x == y).sum())})  p={p:.4g}  -> {'MET' if ok else 'not met'}"
        )
    return out


def main():
    models = ["flat", "Gmix0.5", *VARIANTS]
    pooled_crops = {m: {} for m in models}
    fold_scores = {m: {} for m in models}
    avail_folds = []

    for f in (3, 4, 5, 6, 7):
        paths = {
            "flat": CV5 / f"flat_f{f}/score/latest_official_metrics_extended.json",
            "Gmix0.5": CV5 / f"gh_f{f}/scores/Gmix0.5/latest_official_metrics_extended.json",
            **{v: V2 / f"f{f}/scores/{v}/latest_official_metrics_extended.json" for v in VARIANTS},
        }
        if not all(p.is_file() for p in paths.values()):
            missing = [k for k, p in paths.items() if not p.is_file()]
            print(f"[skip] fold {f} missing: {missing}")
            continue
        avail_folds.append(f)
        loaded = {k: load_metrics(p) for k, p in paths.items()}
        keep = sorted(loaded["flat"])
        for k in models:
            assert set(keep) <= set(loaded[k])
            fold_scores[k][f] = combine({c: loaded[k][c] for c in keep})
            for c in keep:
                pooled_crops[k][f"{f}:{c}"] = loaded[k][c]

    if not avail_folds:
        print("No complete folds scored yet.")
        return

    print(f"5-fold CV summary ({len(pooled_crops['flat'])} crops, folds {avail_folds}):")
    for f in avail_folds:
        parts = [f"{k}={fold_scores[k][f]['overall']:.4f}" for k in models]
        print(f"  fold {f}: " + "  |  ".join(parts))

    pooled = {k: combine(pooled_crops[k]) for k in models}
    print("\nPooled scores:")
    for k in models:
        p = pooled[k]
        print(f"  {k:<22s} overall={p['overall']:.5f}  semantic={p['semantic']:.5f}  instance={p['instance']:.5f}")

    comparisons = {}
    for v in VARIANTS:
        comparisons[f"{v}_vs_Gmix0.5"] = compare_pair("Gmix0.5", pooled_crops["Gmix0.5"], v, pooled_crops[v], fold_scores["Gmix0.5"], fold_scores[v])
        comparisons[f"{v}_vs_flat"] = compare_pair("flat", pooled_crops["flat"], v, pooled_crops[v], fold_scores["flat"], fold_scores[v])

    out_path = V2.parent / "cv5_verdict_v2.json"
    out_path.write_text(json.dumps({
        "folds": avail_folds,
        "crops": len(pooled_crops["flat"]),
        "fold_scores": fold_scores,
        "pooled": pooled,
        "comparisons": comparisons,
    }, indent=2, default=float) + "\n")
    print(f"\nSaved verdict -> {out_path}")


if __name__ == "__main__":
    main()
