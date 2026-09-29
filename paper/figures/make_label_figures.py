"""Label-map figures from the locked CellMap annotations and raw predictions.

Outputs (figures/):
  fig_target_coding.pdf   Fig. 1: one EM window and its flat, binary-conditional and AnchorSlot targets
  fig_multiclass.pdf      Fig. 5: full-slice multi-class comparison (EM / GT / class-balanced flat / AnchorSlot)

Fig. 5 rows: the crop with the most annotated labels (crop219), the median-margin crop (crop126), and
the crop with the largest class-balanced advantage (crop15); central slice along the first array axis.
Fig. 1 uses a window centred on the largest mitochondrion in crop219's central slice. EM comes from the header-corrected Dataset203 images.
Run with the csc environment on a compute node.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cc3d  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

import instance_matches as im  # noqa: E402
from make_result_figures import PREDICTIONS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
DATA = Path("/projects/weilab/liupeng/data/nnUNet/nnUNet_raw")
IMAGES = DATA / "Dataset203_Dataset101_CellMap_model_high_res_hdrfix/imagesTs"
LABEL_IDS = {k: v for k, v in json.loads((DATA / "Dataset200_Dataset101_CellMap_model_high_res/dataset.json").read_text())["labels"].items() if k != "background"}
ANCHORS = ("ecs", "pm", "cyto", "nucpl")
QUERY = ("mito_mem", "mito_lum", "mito_ribo")

# Identity palette: one hue per organelle family, membrane dark and lumen light.
FAMILIES = [
    ("Mitochondrion", {"mito_mem": "#d95f02", "mito_lum": "#fdae6b", "mito_ribo": "#7f2704"}),
    ("ER / ERES", {"er_mem": "#1b7837", "er_lum": "#a6dba0", "eres_mem": "#4d9221", "eres_lum": "#d9f0d3"}),
    ("Golgi", {"golgi_mem": "#6a3d9a", "golgi_lum": "#cab2d6"}),
    ("Vesicle", {"ves_mem": "#e7298a", "ves_lum": "#fbb4d9"}),
    ("Endosome", {"endo_mem": "#8c510a", "endo_lum": "#dfc27d"}),
    ("Lysosome", {"lyso_mem": "#99000d", "lyso_lum": "#fb6a4a"}),
    ("Lipid droplet", {"ld_mem": "#a68c00", "ld_lum": "#fff176"}),
    ("Nuclear env./pore", {"ne_mem": "#2166ac", "ne_lum": "#92c5de", "np_out": "#053061", "np_in": "#4393c3"}),
    ("Chromatin", {"hchrom": "#01665e", "echrom": "#80cdc1"}),
    ("Microtubule", {"mt_out": "#525252", "mt_in": "#a3a3a3"}),
    ("Peroxisome", {"perox_mem": "#4d7c0f", "perox_lum": "#c7e9a0"}),
]
ANCHOR_COLORS = {"ecs": "#dbe6f0", "pm": "#3a3a3a", "cyto": "#f3ead8", "nucpl": "#e3dcef"}
IDENTITY = {**ANCHOR_COLORS, **{k: v for _, fam in FAMILIES for k, v in fam.items()}}
NOT_ANNOTATED = "#8a8a8a"

# Role palette shared with the method figure (Fig. 2).
ROLE = {"bg": "#d9d9d9", "ecs": "#c6dbef", "pm": "#2171b5", "cyto": "#9ecae1", "nucpl": "#6baed6",
        "mito_mem": "#d95f02", "mito_lum": "#fdae6b", "mito_ribo": "#7f2704", "other": "#8e6bbf"}
BINARY = {"negative": "#9e9e9e", "target": "#d95f02"}
INK = "#222222"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42, "axes.linewidth": 0.4})

# (crop, case, row label). Row 1 has the most annotated labels (26 of 31) and is also the crop with the
# largest AnchorSlot - class-balanced margin (rank 1 of 17); rows 2-3 are the median-margin crop (rank 9)
# and the crop with the largest class-balanced advantage (rank 17). Margins: per-crop overall scores in
# cellmap_paper_refresh_20260921/current_report.json.
ROWS = [
    ("crop219", "jrc_sum159-4_crop219", "Crop 219\n(most labels;\nmargin rank 1/17)"),
    ("crop126", "jrc_jurkat-1_crop126", "Crop 126\n(median margin;\nrank 9/17)"),
    ("crop15", "jrc_hela-2_crop15", "Crop 15\n(margin rank\n17/17)"),
]
BINARY_QUERY = "mito_mem"  # the binary baseline queries one of the 31 labels at a time


def annotated_labels(store, crop):
    return [name for name in LABEL_IDS if name in store[crop]]


def gt_label_slice(store, crop, index, labels):
    first = store[crop][labels[0]]
    out = np.zeros(first.shape[1:], dtype=np.int16)
    for name in labels:
        out[np.asarray(store[crop][name][index]) > 0] = LABEL_IDS[name]
    return out, float(first.attrs["voxel_size"][0])


def em_slice(case, index):
    em = np.asarray(nib.load(str(IMAGES / f"{case}_0000.nii.gz")).dataobj[index], dtype=np.float32)
    lo, hi = np.percentile(em, [1, 99])
    return np.clip((em - lo) / (hi - lo + 1e-6), 0, 1)


def pred_label_slice(key, case, index):
    return np.asarray(nib.load(str(Path(PREDICTIONS[key]) / f"{case}.nii.gz")).dataobj[index]).astype(np.int16)


def colour_map(label_map, colours, default, annotated=None):
    rgb = np.empty(label_map.shape + (3,))
    rgb[:] = to_rgb(default)
    for name, idx in LABEL_IDS.items():
        mask = label_map == idx
        if not mask.any() or name not in colours:
            continue
        rgb[mask] = to_rgb(colours[name] if annotated is None or name in annotated else NOT_ANNOTATED)
    return rgb


def overlay(em, label_map, annotated, alpha=0.6):
    rgb = np.repeat(em[..., None], 3, axis=-1)
    colours = colour_map(label_map, IDENTITY, "#000000", annotated)
    mask = (label_map > 0) & (label_map != LABEL_IDS["cyto"])
    rgb[mask] = (1 - alpha) * rgb[mask] + alpha * colours[mask]
    return rgb


def scale_bar(ax, voxel_nm, length_nm, shape):
    h, w = shape
    px = length_nm / voxel_nm
    pad = 0.05 * w
    ax.add_patch(Rectangle((w - pad - px, h - pad - 0.022 * h), px, 0.022 * h, fc="white", ec="black", lw=0.4))


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01, dpi=600)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close(fig)
    print("wrote", FIG / f"{name}.pdf")


def family_handles(present, annotated):
    """Legend: fill = lumen colour, edge = membrane colour, for each family present in the panels."""
    handles = [Patch(fc=ANCHOR_COLORS[a], ec="#666666", lw=0.4, label=a) for a in ("ecs", "pm", "nucpl") if a in present]
    for name, fam in FAMILIES:
        members = [k for k in fam if k in present and k in annotated]
        if members:
            fill = fam[members[1]] if len(members) > 1 else fam[members[0]]
            handles.append(Patch(fc=fill, ec=fam[members[0]], lw=1.4, label=name))
    if any(n not in annotated for n in present):
        handles.append(Patch(fc=NOT_ANNOTATED, ec="none", label="Not annotated in crop"))
    return handles


def multiclass_figure(store):
    fig, axes = plt.subplots(len(ROWS), 4, figsize=(6.3, 1.62 * len(ROWS) + 0.45))
    present, annotated_any, record = set(), set(), []
    for row, (crop, case, label) in enumerate(ROWS):
        labels = annotated_labels(store, crop)
        index = store[crop][labels[0]].shape[0] // 2
        gt, voxel = gt_label_slice(store, crop, index, labels)
        em = em_slice(case, index)
        preds = {k: pred_label_slice(k, case, index) for k in ("cb_flat", "anchorslot")}
        if em.shape != gt.shape or any(p.shape != gt.shape for p in preds.values()):
            raise RuntimeError(f"{crop}: EM/GT/prediction grids differ")
        panels = [("EM", np.repeat(em[..., None], 3, -1)), ("Ground truth", overlay(em, gt, labels)),
                  ("Class-balanced flat", overlay(em, preds["cb_flat"], labels)),
                  ("AnchorSlot", overlay(em, preds["anchorslot"], labels))]
        for col, (title, rgb) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(rgb, interpolation="nearest")
            scale_bar(ax, voxel, 200, gt.shape)
            clean(ax)
            if row == 0:
                ax.set_title(title, fontsize=7.5, pad=2)
        axes[row, 0].set_ylabel(label, fontsize=6.5)
        shown = {n for n in labels if (gt == LABEL_IDS[n]).any()} | {
            n for p in preds.values() for n in LABEL_IDS if (p == LABEL_IDS[n]).any()}
        present |= shown
        annotated_any |= {n for n in shown if n in labels}
        record.append(dict(crop=crop, index=index, voxel_nm=voxel, annotated=len(labels),
                           unannotated_labels_predicted=sorted(n for n in shown if n not in labels)))
    fig.legend(handles=family_handles(present, annotated_any), loc="lower center", bbox_to_anchor=(0.5, -0.06),
               ncol=7, frameon=False, handlelength=1.1, handletextpad=0.35, columnspacing=0.8, fontsize=6.3)
    fig.subplots_adjust(wspace=0.03, hspace=0.04, left=0.1, right=0.995, top=0.95, bottom=0.05)
    save(fig, "fig_multiclass")
    return record


def target_window(gt, half=100):
    mito = np.isin(gt, [LABEL_IDS[q] for q in QUERY])
    comp = cc3d.connected_components(mito.astype(np.uint8)[None])[0]
    ids, sizes = np.unique(comp[comp > 0], return_counts=True)
    cy, cx = np.argwhere(comp == ids[np.argmax(sizes)]).mean(0)
    h, w = gt.shape
    y0 = int(min(max(cy - half, 0), h - 2 * half))
    x0 = int(min(max(cx - half, 0), w - 2 * half))
    return slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half)


def target_coding_figure(store):
    crop, case, _ = ROWS[0]
    labels = annotated_labels(store, crop)
    index = store[crop][labels[0]].shape[0] // 2
    gt_full, voxel = gt_label_slice(store, crop, index, labels)
    region = target_window(gt_full)
    gt, em = gt_full[region], em_slice(case, index)[region]
    flat = colour_map(gt, IDENTITY, ROLE["bg"])
    binary = np.empty(gt.shape + (3,))
    binary[:] = to_rgb(BINARY["negative"])
    binary[gt == LABEL_IDS[BINARY_QUERY]] = to_rgb(BINARY["target"])
    role = colour_map(gt, {k: v for k, v in ROLE.items() if k in LABEL_IDS}, ROLE["bg"])
    dynamic = (gt > 0) & ~np.isin(gt, [LABEL_IDS[n] for n in ANCHORS + QUERY])
    role[dynamic] = to_rgb(ROLE["other"])

    fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.72))
    panels = [("(a) EM", np.repeat(em[..., None], 3, -1)),
              ("(b) Flat (31 labels, one softmax)", flat),
              ("(c) Binary (query: mito_mem)", binary),
              ("(d) AnchorSlot (query: mito group)", role)]
    for ax, (title, rgb) in zip(axes.flat, panels):
        ax.imshow(rgb, interpolation="nearest")
        clean(ax)
        ax.set_title(title, fontsize=6.6, pad=2)
    scale_bar(axes[0, 0], voxel, 200, gt.shape)
    axes[1, 0].legend(handles=[Patch(color=BINARY["target"], label="Target (mito_mem)"),
                               Patch(color=BINARY["negative"], label="Negative (bg, context, mito\nlumen, other organelles)")],
                      loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=6,
                      handlelength=1.0, handletextpad=0.4)
    axes[1, 1].legend(handles=[Patch(color=ROLE["bg"], label="Background"),
                               Patch(color=ROLE["cyto"], label="Anchors (4)"),
                               Patch(color=ROLE["mito_mem"], label="Active slots"),
                               Patch(color=ROLE["other"], label="Non-query FG")],
                      loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=6,
                      handlelength=1.0, handletextpad=0.4, columnspacing=0.6)
    fig.subplots_adjust(wspace=0.04, hspace=0.12, left=0.005, right=0.995, top=0.95, bottom=0.13)
    save(fig, "fig_target_coding")
    return dict(crop=crop, index=index, window=[[s.start, s.stop] for s in region], voxel_nm=voxel,
                binary_query=BINARY_QUERY)


def main():
    store = im.truth_store()
    record = {"image_source": str(IMAGES),
              "selection": ("rows: most annotated labels (crop219, also margin rank 1), median AnchorSlot-minus-CB "
                            "margin (crop126, rank 9), largest CB advantage (crop15, rank 17); central slice on axis 0"),
              "multiclass": multiclass_figure(store),
              "target_coding": target_coding_figure(store),
              "predictions": {k: PREDICTIONS[k] for k in ("cb_flat", "anchorslot")}}
    (FIG / "label_figures_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("multiclass", "target_coding")}))


if __name__ == "__main__":
    main()
