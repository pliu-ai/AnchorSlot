"""Fig. 1: CellMap data challenge and supervision roles.

(a) 3D rendering of one annotated crop (renders/<case>_v0a20.png from render_crop.py)
(b) share of annotated voxels per label over all labeled crops (crop_scan.json from scan_crops.py)
(c)-(f) one EM window of the same crop and its flat, binary-conditional and AnchorSlot targets
Run with the csc environment:  python make_fig01.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cc3d  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
RAW = Path("/projects/weilab/liupeng/data/nnUNet/nnUNet_raw/Dataset205_Dataset101_CellMap_model_high_res_truegeom")
CASE = "jrc_sum159-4_crop219"
RENDER = HERE / "renders" / f"{CASE}_v0a20.png"
LABEL_IDS = {k: v for k, v in json.loads((RAW / "dataset.json").read_text())["labels"].items() if k != "background"}
ANCHORS = ("ecs", "pm", "cyto", "nucpl")
QUERY = ("mito_mem", "mito_lum", "mito_ribo")
BINARY_QUERY = "mito_mem"

IDENTITY = {
    "ecs": "#dbe6f0", "pm": "#3a3a3a", "cyto": "#f3ead8", "nucpl": "#e3dcef",
    "mito_mem": "#d95f02", "mito_lum": "#fdae6b", "mito_ribo": "#7f2704",
    "er_mem": "#1b7837", "er_lum": "#a6dba0", "eres_mem": "#4d9221", "eres_lum": "#d9f0d3",
    "golgi_mem": "#6a3d9a", "golgi_lum": "#cab2d6", "ves_mem": "#e7298a", "ves_lum": "#fbb4d9",
    "endo_mem": "#8c510a", "endo_lum": "#dfc27d", "lyso_mem": "#99000d", "lyso_lum": "#fb6a4a",
    "ld_mem": "#a68c00", "ld_lum": "#fff176", "ne_mem": "#2166ac", "ne_lum": "#92c5de",
    "np_out": "#053061", "np_in": "#4393c3", "hchrom": "#01665e", "echrom": "#80cdc1",
    "mt_out": "#525252", "mt_in": "#a3a3a3", "perox_mem": "#4d7c0f", "perox_lum": "#c7e9a0",
}
ROLE = {"bg": "#d9d9d9", "ecs": "#c6dbef", "pm": "#2171b5", "cyto": "#9ecae1", "nucpl": "#6baed6",
        "mito_mem": "#d95f02", "mito_lum": "#fdae6b", "mito_ribo": "#7f2704", "other": "#8e6bbf"}
BINARY = {"negative": "#9e9e9e", "target": "#d95f02"}
ANCHOR_INK = "#2171b5"
INK = "#222222"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6, "pdf.fonttype": 42, "axes.linewidth": 0.4})

# Direct labels on the rendering: text, text position and (optional) pointed position, as fractions of the image.
CALLOUTS = [
    ("Nucleus", (0.13, 0.36), None),
    ("ER", (0.33, 0.215), None),
    ("Lipid droplet", (0.17, 0.045), (0.385, 0.105)),
    ("Mitochondrion", (0.80, 0.055), (0.66, 0.075)),
    ("Endosome", (0.505, 0.525), None),
    ("Lysosome", (0.19, 0.50), (0.41, 0.42)),
    ("Vesicles", (0.12, 0.665), (0.225, 0.722)),
    ("Golgi", (0.37, 0.87), None),
]


def render_panel(ax) -> dict:
    image = Image.open(RENDER).convert("RGB")
    arr = np.asarray(image)
    ys, xs = np.where((arr < 250).any(2))
    pad = 12
    box = (xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad + 70)
    image = image.crop(box)
    ax.imshow(image, interpolation="lanczos")
    h, w = image.size[1], image.size[0]
    outline = [pe.withStroke(linewidth=1.6, foreground="white")]
    for text, (tx, ty), target in CALLOUTS:
        if target is None:
            ax.text(tx * w, ty * h, text, ha="center", va="center", fontsize=6, color=INK, weight="bold",
                    path_effects=outline, linespacing=0.95)
        else:
            ax.annotate(text, xy=(target[0] * w, target[1] * h), xytext=(tx * w, ty * h), ha="center", va="center",
                        fontsize=6, color=INK, weight="bold", path_effects=outline,
                        arrowprops=dict(arrowstyle="-", lw=0.6, color=INK, shrinkA=1, shrinkB=0))
    meta = json.loads(RENDER.with_suffix(".json").read_text())
    edge_um = meta["size_nm"][0] / 1000
    ax.text(0.36 * w, 0.972 * h, f"{edge_um:.1f} µm", ha="center", va="center", fontsize=6, color=INK, rotation=-6.5)
    ax.set_axis_off()
    return {"render": str(RENDER), "crop_box": [int(v) for v in box], **meta}


def imbalance_panel(ax) -> dict:
    scan = json.loads((HERE / "crop_scan.json").read_text())
    names = {i: n for n, i in LABEL_IDS.items()}
    volume = {name: 0 for name in LABEL_IDS}
    total = 0
    for crop in scan["crops"].values():  # voxels on each crop's annotation grid; 255 = not annotated
        for key, count in crop["counts"].items():
            if int(key) == 255:
                continue
            total += count
            if int(key) > 0:
                volume[names[int(key)]] += count
    share = {n: 100 * v / total for n, v in volume.items()}
    order = sorted(share, key=share.get, reverse=True)
    y = np.arange(len(order))
    ax.barh(y, [share[n] for n in order], color=[IDENTITY[n] for n in order], edgecolor="#555555", linewidth=0.25,
            height=0.78)
    ax.set_xscale("log")
    ax.set_xlim(0.006, 100)
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=4.6)
    for tick, name in zip(ax.get_yticklabels(), order):
        if name in ANCHORS:
            tick.set_color(ANCHOR_INK)
            tick.set_fontweight("bold")
    ax.set_xticks([0.01, 0.1, 1, 10, 100])
    ax.set_xticklabels(["0.01", "0.1", "1", "10", "100"], fontsize=5.5)
    ax.set_xlabel("Share of annotated voxels (%)", fontsize=6, labelpad=1.5)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.tick_params(axis="x", length=2, pad=1.5, width=0.4)
    ax.tick_params(axis="x", which="minor", length=1.2, width=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    anchors = sum(share[n] for n in ANCHORS)
    ratio = share[order[0]] / share[order[-1]]
    organelles = sum(share.values()) - anchors
    note = dict(transform=ax.transAxes, ha="right", va="center", fontsize=5.6, linespacing=1.05)
    ax.text(0.995, 0.50, f"4 context labels\n{anchors:.0f}% of voxels", color=ANCHOR_INK, weight="bold", **note)
    ax.text(0.995, 0.335, f"27 organelle\nsubstructures\n{organelles:.0f}%", color=INK, **note)
    ax.text(0.995, 0.13, f"largest / smallest\n≈ {round(ratio, -2):,.0f}×", color=INK, **note)
    return {"share_percent": {n: share[n] for n in order}, "anchor_share": anchors, "organelle_share": organelles,
            "ratio": ratio,
            "n_crops": len(scan["crops"])}


def target_window(gt: np.ndarray, half: int = 100) -> tuple[slice, slice]:
    mito = np.isin(gt, [LABEL_IDS[q] for q in QUERY])
    comp = cc3d.connected_components(mito.astype(np.uint8)[None])[0]
    ids, sizes = np.unique(comp[comp > 0], return_counts=True)
    cy, cx = np.argwhere(comp == ids[np.argmax(sizes)]).mean(0)
    h, w = gt.shape
    y0 = int(min(max(cy - half, 0), h - 2 * half))
    x0 = int(min(max(cx - half, 0), w - 2 * half))
    return slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half)


def colour_map(label_map: np.ndarray, colours: dict[str, str], default: str) -> np.ndarray:
    rgb = np.empty(label_map.shape + (3,))
    rgb[:] = to_rgb(default)
    for name, idx in LABEL_IDS.items():
        if name in colours:
            rgb[label_map == idx] = to_rgb(colours[name])
    return rgb


def target_panels(axes) -> dict:
    label_nii = nib.load(RAW / "labelsTr" / f"{CASE}.nii.gz")
    index = label_nii.shape[0] // 2
    gt_full = np.asarray(label_nii.dataobj[index]).astype(np.int16)
    em_full = np.asarray(nib.load(RAW / "imagesTr" / f"{CASE}_0000.nii.gz").dataobj[index], dtype=np.float32)
    lo, hi = np.percentile(em_full, [1, 99])
    em_full = np.clip((em_full - lo) / (hi - lo + 1e-6), 0, 1)
    region = target_window(gt_full)
    gt, em = gt_full[region], em_full[region]
    voxel = float(label_nii.header.get_zooms()[1])
    flat = colour_map(gt, IDENTITY, ROLE["bg"])
    binary = np.empty(gt.shape + (3,))
    binary[:] = to_rgb(BINARY["negative"])
    binary[gt == LABEL_IDS[BINARY_QUERY]] = to_rgb(BINARY["target"])
    role = colour_map(gt, {k: v for k, v in ROLE.items() if k in LABEL_IDS}, ROLE["bg"])
    dynamic = (gt > 0) & ~np.isin(gt, [LABEL_IDS[n] for n in ANCHORS + QUERY])
    role[dynamic] = to_rgb(ROLE["other"])
    panels = [("(c) EM", np.repeat(em[..., None], 3, -1)), ("(d) Flat target", flat),
              ("(e) Binary target", binary), ("(f) AnchorSlot target", role)]
    for ax, (title, rgb) in zip(axes, panels):
        ax.imshow(rgb, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=6.2, pad=2)
    h, w = gt.shape
    px = 200 / voxel
    axes[0].add_patch(Rectangle((w - 0.05 * w - px, h - 0.05 * w - 0.022 * h), px, 0.022 * h, fc="white", ec="black", lw=0.4))
    legend = dict(loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=5.3, handlelength=0.9,
                  handletextpad=0.35, columnspacing=0.5, borderaxespad=0.2, labelspacing=0.25)
    axes[2].legend(handles=[Patch(color=BINARY["target"], label="Target"),
                            Patch(color=BINARY["negative"], label="Everything else")], ncol=1, **legend)
    axes[3].legend(handles=[Patch(color=ROLE["bg"], label="Background"), Patch(color=ROLE["cyto"], label="Anchors"),
                            Patch(color=ROLE["mito_mem"], label="Active slots"),
                            Patch(color=ROLE["other"], label="Non-query FG")], ncol=2, **legend)
    return {"case": CASE, "axis0_index": index, "window": [[s.start, s.stop] for s in region], "voxel_nm": voxel}


def main() -> None:
    fig = plt.figure(figsize=(7.16, 2.86))
    ax_a = fig.add_axes([0.0, 0.005, 0.355, 0.925])
    ax_b = fig.add_axes([0.432, 0.108, 0.205, 0.822])
    inner = fig.add_gridspec(2, 2, left=0.668, right=0.99, top=0.93, bottom=0.108, wspace=0.06, hspace=0.17)
    axes = [fig.add_subplot(inner[i, j]) for i in range(2) for j in range(2)]
    record = {"a": render_panel(ax_a), "b": imbalance_panel(ax_b), "c_f": target_panels(axes)}
    ax_a.set_title("(a) One annotated CellMap crop", fontsize=6.8, pad=2)
    ax_b.set_title("(b) Label imbalance", fontsize=6.8, pad=2)
    out = HERE / "fig_cellmap_roles.pdf"
    fig.savefig(out, dpi=500)
    fig.savefig(out.with_suffix(".png"), dpi=260)
    (HERE / "fig_cellmap_roles_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
