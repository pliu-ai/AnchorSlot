"""Thumbnails for Fig. 2(a): one EM window of crop219 and its role-coded targets under two group queries."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from matplotlib.colors import to_rgb
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = Path("/projects/weilab/liupeng/papers/AnchorSlot_overleaf/figures")
RAW = Path("/projects/weilab/liupeng/data/nnUNet/nnUNet_raw/Dataset205_Dataset101_CellMap_model_high_res_truegeom")
import os
CASE, SIZE = "jrc_sum159-4_crop219", 160
AXIS, INDEX, Y0, X0 = (int(v) for v in os.environ.get("FIG02_WINDOW", "1,32,240,0").split(","))
L = json.loads((RAW / "dataset.json").read_text())["labels"]
ROLE = {"bg": "#d9d9d9", "ecs": "#c6dbef", "pm": "#2171b5", "cyto": "#9ecae1", "nucpl": "#6baed6", "other": "#8e6bbf"}
SLOTS = ("#d95f02", "#fdae6b", "#7f2704")
QUERIES = {"mito": ("mito_mem", "mito_lum", "mito_ribo"), "golgi": ("golgi_mem", "golgi_lum")}
SCALE = 4  # nearest-neighbour upsampling of the saved PNGs

window = (slice(Y0, Y0 + SIZE), slice(X0, X0 + SIZE))
label_nii = nib.load(RAW / "labelsTr" / f"{CASE}.nii.gz")
gt = np.take(np.asarray(label_nii.dataobj), INDEX, axis=AXIS)[window]
em_full = np.take(np.asarray(nib.load(RAW / "imagesTr" / f"{CASE}_0000.nii.gz").dataobj), INDEX, axis=AXIS)
em = em_full[window].astype(np.float32)
lo, hi = np.percentile(em, [0.5, 99.5])  # contrast stretched within the window
em = np.clip((em - lo) / (hi - lo + 1e-6), 0, 1)
voxel_nm = float(label_nii.header.get_zooms()[0])


def save(rgb: np.ndarray, name: str) -> None:
    image = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    image.resize((SIZE * SCALE, SIZE * SCALE), Image.NEAREST).save(OUT / name)


em_rgb = np.repeat(em[..., None], 3, -1)
bar_px = int(round(200 / voxel_nm))  # 200 nm scale bar
em_rgb[SIZE - 12:SIZE - 8, SIZE - 8 - bar_px:SIZE - 8] = 1.0
save(em_rgb, "fig02_em.png")
for query, members in QUERIES.items():
    rgb = np.empty(gt.shape + (3,))
    rgb[:] = to_rgb(ROLE["bg"])
    rgb[gt > 0] = to_rgb(ROLE["other"])
    for anchor in ("ecs", "pm", "cyto", "nucpl"):
        rgb[gt == L[anchor]] = to_rgb(ROLE[anchor])
    for name, colour in zip(members, SLOTS):
        rgb[gt == L[name]] = to_rgb(colour)
    save(rgb, f"fig02_role_{query}.png")
present = sorted(n for n, i in L.items() if (gt == i).any())
(HERE / "fig02_thumbs_provenance.json").write_text(json.dumps({
    "case": CASE, "axis": AXIS, "index": INDEX, "window": [[Y0, Y0 + SIZE], [X0, X0 + SIZE]], "voxel_nm": voxel_nm,
    "window_nm": SIZE * voxel_nm, "scale_bar_nm": 200, "labels_present": present}, indent=2) + "\n")
print(present, voxel_nm, flush=True)
