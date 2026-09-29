"""Per-label CellMap Dice figure (flat, class-balanced flat, AnchorSlot).

Reads the pooled-count Dice values from tables/cellmap_per_class_current.tex so the
figure and the table cannot drift apart, and writes figures/fig_per_label_dice.pdf.
Labels are ordered as in Table I: fixed anchors first, then dynamic groups G1-G13.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "tables" / "cellmap_per_class_current.tex"
OUT = ROOT / "figures" / "fig_per_label_dice.pdf"

GROUPS = [
    ("Anchors", ["ecs", "pm", "cyto", "nucpl"]),
    ("Mito", ["mito_mem", "mito_lum", "mito_ribo"]),
    ("Golgi", ["golgi_mem", "golgi_lum"]),
    ("Ves", ["ves_mem", "ves_lum"]),
    ("Endo", ["endo_mem", "endo_lum"]),
    ("Lyso", ["lyso_mem", "lyso_lum"]),
    ("LD", ["ld_mem", "ld_lum"]),
    ("ERES", ["eres_mem", "eres_lum"]),
    ("Chrom", ["hchrom", "echrom"]),
    ("NE", ["ne_mem", "ne_lum"]),
    ("NP", ["np_out", "np_in"]),
    ("MT", ["mt_out", "mt_in"]),
    ("Perox", ["perox_mem", "perox_lum"]),
    ("ER", ["er_mem", "er_lum"]),
]

# Categorical slots 1-3 of the validated reference palette (all-pairs pass, light mode);
# marker shape is the secondary encoding for print and colour-vision deficiency.
SERIES = [
    ("AnchorSlot", 3, "#2a78d6", "o", True),
    ("Class-balanced flat", 2, "#eb6834", "^", True),
    ("Flat nnU-Net", 1, "#1baf7a", "s", False),
]

ROW = re.compile(r"^\\texttt\{(?P<label>[a-z_\\]+)\}\s*&(?P<vals>.*)\\\\\s*$")


def read_table(path: Path) -> dict[str, list[float]]:
    rows: dict[str, list[float]] = {}
    for line in path.read_text().splitlines():
        match = ROW.match(line.strip())
        if not match:
            continue
        label = match.group("label").replace("\\_", "_")
        cells = [re.sub(r"\\textbf\{([^}]*)\}", r"\1", c).strip() for c in match.group("vals").split("&")]
        rows[label] = [float(c) for c in cells]
    return rows


def main() -> None:
    values = read_table(TABLE)
    order = [label for _, labels in GROUPS for label in labels]
    missing = [label for label in order if label not in values]
    if missing or len(values) != 31:
        raise SystemExit(f"table parse mismatch: missing={missing}, parsed={len(values)}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 6.5,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "pdf.fonttype": 42,
    })
    fig, (ax, dax) = plt.subplots(2, 1, figsize=(7.16, 2.55), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.15], "hspace": 0.08})
    x_of = {label: i for i, label in enumerate(order)}

    # Group separators and group names above the plot.
    start = 0
    for name, labels in GROUPS:
        end = start + len(labels)
        if start > 0:
            ax.axvline(start - 0.5, color="#d9d8d4", lw=0.5, zorder=0)
        ax.text((start + end - 1) / 2, 103, name, ha="center", va="bottom", fontsize=6, color="#52514e")
        start = end
    ax.axvspan(-0.5, 3.5, color="#f0efec", zorder=-1, lw=0)

    offsets = {"AnchorSlot": 0.22, "Class-balanced flat": 0.0, "Flat nnU-Net": -0.22}
    for name, col, color, marker, filled in SERIES:
        xs = [x_of[label] + offsets[name] for label in order]
        ys = [values[label][col - 1] for label in order]
        ax.scatter(
            xs, ys, s=11, marker=marker, linewidths=0.7, zorder=3,
            facecolors=color if filled else "#fcfcfb", edgecolors=color, label=name,
        )

    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(-3, 101)
    ax.set_ylabel("Dice (%)")
    # Lower strip: AnchorSlot minus class-balanced flat, coloured by the better method.
    deltas = [values[label][2] - values[label][1] for label in order]
    dax.bar(range(len(order)), deltas, width=0.6, zorder=3,
            color=["#2a78d6" if d >= 0 else "#eb6834" for d in deltas])
    dax.axhline(0, color="#52514e", lw=0.5, zorder=2)
    dax.axvspan(-0.5, 3.5, color="#f0efec", zorder=-1, lw=0)
    for boundary in np.cumsum([len(labels) for _, labels in GROUPS])[:-1]:
        dax.axvline(boundary - 0.5, color="#d9d8d4", lw=0.5, zorder=0)
    dax.set_ylim(-35, 20)
    dax.set_yticks([-30, 0, 15])
    dax.set_ylabel("$\\Delta$ vs CB\n(pp)", fontsize=6)
    dax.grid(axis="y", color="#e6e5e1", lw=0.4, zorder=0)
    for side in ("top", "right"):
        dax.spines[side].set_visible(False)
    dax.tick_params(axis="x", length=0, pad=2)
    dax.set_xticks(range(len(order)))
    dax.set_xticklabels(order, rotation=90, fontfamily="DejaVu Sans Mono", fontsize=6)
    ax.tick_params(axis="x", labelbottom=False)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color="#e6e5e1", lw=0.4, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.09), ncol=3, frameon=False,
              handletextpad=0.3, columnspacing=1.2, fontsize=6.5)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.01)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
