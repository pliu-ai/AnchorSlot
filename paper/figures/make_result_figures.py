"""Render the CellMap result figures from raw predictions and the locked native annotations.

Outputs (figures/):
  fig_instance_crop219.pdf  instance matching in crop219 (class-balanced flat vs AnchorSlot)
  fig_qualitative.pdf       fixed qualitative locations, flat / class-balanced flat / AnchorSlot
  fig_failure.pdf           AnchorSlot failure cases

Cases and slices are the fixed selections of the earlier figures
(fig03_qualitative_selection_metadata.json and the failure list below); only the layout,
crops of the two sparse failure rows, legends, and scale bars change. Instance matches come
from the pinned evaluator via instance_matches.py and are checked against its stored counts.
Run with the csc environment (zarr, nibabel, cc3d) on a compute node.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Patch, Rectangle  # noqa: E402

import instance_matches as im  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RUNS = Path("/projects/weilab/liupeng/runs/Anchorslot")
IMAGES = Path("/projects/weilab/liupeng/data/nnUNet/nnUNet_raw/Dataset125_CellMap_Res2-4nm_ralphUNet/imagesTs")
PREDICTIONS = {
    key: json.loads((RUNS / "evaluation_matrix" / sub / "full_annotated/4nm/evaluation_provenance.json").read_text())["predictions"]
    for key, sub in {
        "flat": "table_vi_current/flat_nnunet",
        "cb_flat": "flat_nnunet_class_balanced",
        "anchorslot": "anchorslot_primary",
    }.items()
}

# Overlay colours of the earlier figures (CVD-checked: deutan dE 8.2 for FP vs TP).
TP = np.array([0.12, 0.78, 0.28])
FP = np.array([1.00, 0.12, 0.45])
FN = np.array([0.00, 0.70, 1.00])
GT_FILL = np.array([1.00, 0.78, 0.10])
INK = "#222222"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42, "axes.linewidth": 0.4})


def plane(array, axis, index):
    region = [slice(None)] * 3
    region[axis] = index
    return np.asarray(array[tuple(region)])


def em_slice(case, axis, index):
    out = plane(nib.load(str(IMAGES / f"{case}_0000.nii.gz")).dataobj, axis, index).astype(np.float32)
    lo, hi = np.percentile(out, [1, 99])
    return np.clip((out - lo) / (hi - lo + 1e-6), 0, 1)


def pred_slice(key, case, axis, index):
    return plane(nib.load(str(Path(PREDICTIONS[key]) / f"{case}.nii.gz")).dataobj, axis, index)


def truth_slice(store, case, label, axis, index):
    crop = "crop" + case.rsplit("_crop", 1)[1]
    arr = store[crop][label]
    return plane(arr, axis, index) > 0, float(arr.attrs["voxel_size"][0])


def overlay(em, gt, pred=None):
    rgb = np.repeat(em[..., None], 3, axis=-1)
    if pred is None:
        rgb[gt] = 0.3 * rgb[gt] + 0.7 * GT_FILL
        return rgb
    for mask, color in ((gt & pred, TP), (~gt & pred, FP), (gt & ~pred, FN)):
        rgb[mask] = 0.2 * rgb[mask] + 0.8 * color
    return rgb


def scale_bar(ax, voxel_nm, length_nm, shape, color="white"):
    """Bottom-right scale bar of length_nm, drawn in data (pixel) coordinates."""
    h, w = shape
    px = length_nm / voxel_nm
    pad = 0.06 * w
    edge = "black" if color == "white" else "none"
    ax.add_patch(Rectangle((w - pad - px, h - pad - 0.025 * h), px, 0.025 * h, fc=color, ec=edge, lw=0.4))


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def legend(fig, handles, y):
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(handles),
               frameon=False, handlelength=1.0, handletextpad=0.4, columnspacing=0.9, fontsize=6.5)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01, dpi=600)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close(fig)
    print("wrote", FIG / f"{name}.pdf")


SLAB = 40  # slices per displayed slab (160 nm at 4 nm)


def unmatched_slab_start(depth, models):
    """Slab start covering the most unmatched components, summed over all models.

    The rule is model-neutral: every unmatched component of every model counts once.
    """
    from scipy import ndimage

    diff = np.zeros(depth + 1, dtype=np.int64)
    for pred_cc, matched in models:
        for label, box in enumerate(ndimage.find_objects(pred_cc), start=1):
            if box is None or label in matched:
                continue
            lo = max(0, box[0].start - SLAB + 1)
            hi = min(depth - SLAB, box[0].stop - 1)
            if lo <= hi:
                diff[lo] += 1
                diff[hi + 1] -= 1
    coverage = np.cumsum(diff)[: depth - SLAB + 1]
    return int(np.argmax(coverage)), int(coverage.max())


def instance_figure(store):
    crop, case = "crop219", "jrc_sum159-4_crop219"
    classes = [("lyso", "Lysosome"), ("mito", "Mitochondrion"), ("ld", "Lipid droplet")]
    rows = [("cb_flat", "Class-balanced flat"), ("anchorslot", "AnchorSlot")]
    em_volume = np.asarray(nib.load(str(IMAGES / f"{case}_0000.nii.gz")).dataobj, dtype=np.float32)
    fig, axes = plt.subplots(2, 3, figsize=(3.45, 2.55))
    record = {}
    for col, (cls, title) in enumerate(classes):
        results = {model: im.match(model, crop, cls) for model, _ in rows}
        truth = results[rows[0][0]][0]
        if em_volume.shape != truth.shape:
            raise RuntimeError(f"EM {em_volume.shape} and truth {truth.shape} grids differ")
        start, covered = unmatched_slab_start(truth.shape[0], [(r[1], r[2]) for r in results.values()])
        slab = slice(start, start + SLAB)
        em = em_volume[start + SLAB // 2]
        lo, hi = np.percentile(em, [1, 99])
        em = np.clip((em - lo) / (hi - lo + 1e-6), 0, 1)
        gt_proj = (truth[slab] > 0).any(0)
        for row, (model, name) in enumerate(rows):
            _, pred_cc, matched, counts, voxel = results[model]
            record[f"{model}/{cls}"] = dict(counts, slab=[start, start + SLAB], unmatched_in_slab_both_models=covered)
            sub = pred_cc[slab]
            matched_mask = np.isin(sub, sorted(matched))
            fp_mask = (sub > 0) & ~matched_mask
            rgb = np.repeat(em[..., None], 3, axis=-1)
            for proj, color, keep in ((gt_proj, GT_FILL, 0.45), (matched_mask.any(0), TP, 0.25), (fp_mask.any(0), FP, 0.2)):
                rgb[proj] = keep * rgb[proj] + (1 - keep) * color
            ax = axes[row, col]
            ax.imshow(rgb, interpolation="nearest")
            zz, yy, xx = np.nonzero(fp_mask)
            labels = sub[zz, yy, xx]
            for fid in np.unique(labels):
                sel = labels == fid
                ax.add_patch(Circle((xx[sel].mean(), yy[sel].mean()), 10, fill=False, ec=FP, lw=0.6))
            ax.text(0.03, 0.04, f"TP {counts['tp']}  FP {counts['fp']}", transform=ax.transAxes, va="bottom",
                    ha="left", fontsize=6, color=INK, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
            scale_bar(ax, voxel[0], 500, truth.shape[1:])
            clean(ax)
            if row == 0:
                ax.set_title(title, fontsize=7, pad=2)
            if col == 0:
                ax.set_ylabel(name, fontsize=7)
    legend(fig, [Patch(color=GT_FILL, label="Ground truth"), Patch(color=TP, label="Matched prediction"),
                 Patch(color=FP, label="Unmatched (circled)")], -0.045)
    fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0.07, right=0.995, top=0.93, bottom=0.05)
    save(fig, "fig_instance_crop219")
    return record


def qualitative_figure(store):
    meta = json.loads((FIG / "fig03_qualitative_selection_metadata.json").read_text())
    columns = [k for k in meta if k != "Vesicle/endosome"]
    rows = [("flat", "Flat nnU-Net"), ("cb_flat", "Class-balanced flat"), ("anchorslot", "AnchorSlot")]
    fig, axes = plt.subplots(3, 3, figsize=(3.45, 3.62))
    for col, title in enumerate(columns):
        m = meta[title]
        case, axis, index = m["case"], m["slice_axis"], m["slice_index"]
        region = (slice(*m["crop_y"]), slice(*m["crop_x"]))
        em = em_slice(case, axis, index)[region]
        gt, voxel = truth_slice(store, case, m["label"], axis, index)
        gt = gt[region]
        for row, (key, name) in enumerate(rows):
            pred = pred_slice(key, case, axis, index)[region] == m["label_value"]
            ax = axes[row, col]
            ax.imshow(overlay(em, gt, pred), interpolation="nearest")
            scale_bar(ax, voxel, 100, gt.shape)
            clean(ax)
            if row == 0:
                ax.set_title(title.replace(" ", "\n", 1), fontsize=7, pad=2)
            if col == 0:
                ax.set_ylabel(name, fontsize=7)
    legend(fig, [Patch(color=TP, label="True positive"), Patch(color=FP, label="False positive"),
                 Patch(color=FN, label="False negative")], -0.035)
    fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0.07, right=0.995, top=0.93, bottom=0.04)
    save(fig, "fig_qualitative")


def square_region(center, half, shape):
    region = []
    for c, s in zip(center, shape):
        lo = int(min(max(c - half, 0), max(s - 2 * half, 0)))
        region.append(slice(lo, min(s, lo + 2 * half)))
    return tuple(region)


def failure_figure(store):
    # (case, slice index on axis 0, label id, label, row title, crop mode). Same slices as before;
    # sparse rows are re-centred on their largest ground-truth object with a fixed half-width.
    cases = [
        ("jrc_mus-liver-zon-1_crop266", 390, 6, "mito_ribo", "Mito. ribosome", ("largest", 45)),
        ("jrc_mus-liver-zon-1_crop266", 104, 26, "echrom", "Euchromatin", ("largest", 55)),
        ("jrc_sum159-1_crop20", 57, 15, "ld_mem", "Lipid-droplet\nmembrane", ("bbox", 30)),
        ("jrc_fly-mb-1a_crop121", 229, 9, "ves_mem", "Vesicle\nmembrane", ("largest", 50)),
    ]
    import cc3d

    fig, axes = plt.subplots(4, 3, figsize=(3.45, 4.62))
    saved = []
    for row, (case, index, label_id, label, title, (mode, param)) in enumerate(cases):
        em = em_slice(case, 0, index)
        gt, voxel = truth_slice(store, case, label, 0, index)
        pred = pred_slice("anchorslot", case, 0, index) == label_id
        if mode == "largest":
            comp = cc3d.connected_components(gt.astype(np.uint8)[None])[0]
            ids, sizes = np.unique(comp[comp > 0], return_counts=True)
            center = np.argwhere(comp == ids[np.argmax(sizes)]).mean(0)
            half = param
        else:
            pts = np.argwhere(gt)
            lo, hi = pts.min(0), pts.max(0)
            center = (lo + hi) / 2
            half = max(60, int(max(hi - lo) // 2 + param))
        region = square_region(center, half, gt.shape)
        em_c, gt_c, pred_c = em[region], gt[region], pred[region]
        for col, rgb in enumerate((np.repeat(em_c[..., None], 3, -1), overlay(em_c, gt_c), overlay(em_c, gt_c, pred_c))):
            ax = axes[row, col]
            ax.imshow(rgb, interpolation="nearest")
            scale_bar(ax, voxel, 100, gt_c.shape)
            clean(ax)
        axes[row, 0].set_ylabel(title, fontsize=7)
        saved.append(dict(case=case, axis=0, index=index, label=label, voxel_nm=voxel,
                          crop=[[s.start, s.stop] for s in region], gt_pixels_in_crop=int(gt_c.sum()),
                          gt_voxels_in_slice=int(gt.sum())))
    for col, title in enumerate(("EM", "Ground truth", "AnchorSlot")):
        axes[0, col].set_title(title, fontsize=7, pad=2)
    legend(fig, [Patch(color=GT_FILL, label="Ground truth"), Patch(color=TP, label="TP"),
                 Patch(color=FP, label="FP"), Patch(color=FN, label="FN")], -0.03)
    fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0.1, right=0.995, top=0.95, bottom=0.035)
    save(fig, "fig_failure")
    return saved


def main():
    store = im.truth_store()
    record = {
        "postprocessing": False,
        "instance_crop219": instance_figure(store),
        "failure_examples": failure_figure(store),
        "qualitative_selection": "figures/fig03_qualitative_selection_metadata.json (fixed, not reselected)",
        "predictions": PREDICTIONS,
    }
    qualitative_figure(store)
    (FIG / "result_figures_provenance.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
