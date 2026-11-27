"""Inference-only decoders for AnchorSlot group-head dumps (ideas 1 and 4 of the 2026-10-13 semantic proposals).

All score maps are 32 x Z x Y x X in original CellMap label order; unused labels hold LOW. Inputs follow the dump
layout of predict_dump_gh.py (channels.json): mean background, 4 mean anchors, then per group its slot probabilities
and its 'other' probability, then (optionally) 18 group-head probabilities.

Idea 1, marginal-consistent decoding (`marginal_scores`): the dynamic-foreground mass averaged over the 13 queries,
    1 - P(bg) - sum_a P(a) = mean_g (sum of g's slots + P_g(other)),
is shared out over all dynamic labels in proportion to their slot probabilities:
    M_c = P(D) * P_{g(c),s(c)} / sum_c' P_{g(c'),s(c')}.
Background and anchors are unchanged, and the order among dynamic labels is unchanged.

Idea 4, hysteresis reconstruction (`scale_dynamic` + `hysteresis`): the core segmentation is the unchanged decode; the
relaxed one multiplies every dynamic score by alpha > 1. Relaxed voxels of label c replace background/anchor core
voxels only when they connect (6-connectivity) to a core component of c, so no new component of any label appears.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
from scipy import ndimage

LOW = -1e4          # same filler as the frozen compact_scores
EPS = 1e-8
DEFAULT_DUMP_META = Path(__file__).parent / "channels.json"


def _slots(meta: Sequence[dict]) -> list:
    return [c for c in meta if c["kind"] == "slot"]


def _anchors(meta: Sequence[dict]) -> list:
    return sorted((c for c in meta if c["kind"] == "anchor_mean"), key=lambda c: int(c["index"]))


def dynamic_labels(meta: Sequence[dict]) -> list:
    return [int(c["label"]) for c in _slots(meta)]


def static_labels(meta: Sequence[dict]) -> list:
    return [0, *[int(c["label"]) for c in _anchors(meta)]]


def legacy_scores(p: np.ndarray, meta: Sequence[dict]) -> np.ndarray:
    """Original merge: mean background/anchors; each dynamic label from its own query run."""
    s = np.full((32, *p.shape[1:]), LOW, dtype=np.float32)
    s[0] = p[0]
    for c in meta:
        if c["kind"] in ("anchor_mean", "slot"):
            s[int(c["label"])] = p[int(c["index"])]
    return s


def marginal_scores(p: np.ndarray, meta: Sequence[dict], eps: float = EPS) -> np.ndarray:
    """Idea 1: legacy scores with each dynamic label replaced by P(D) * its share of the summed slot probabilities."""
    s = legacy_scores(p, meta)
    anchor_idx = [int(c["index"]) for c in _anchors(meta)]
    mass = np.clip(1.0 - p[0] - p[anchor_idx].sum(axis=0), 0.0, 1.0)
    slots = _slots(meta)
    total = np.zeros(p.shape[1:], dtype=np.float32)
    for c in slots:
        total += p[int(c["index"])]
    scale = mass / np.maximum(total, eps)
    for c in slots:
        s[int(c["label"])] = p[int(c["index"])] * scale
    return s


def gmix_scores(s: np.ndarray, p: np.ndarray, meta: Sequence[dict], head: np.ndarray, weight: float = 0.5,
                eps: float = EPS) -> np.ndarray:
    """Gmix<w> of the frozen merge, with the structured part `s` passed in (legacy or marginal scores).

    out = exp((1-w) log s + w log H), H = head(bg/anchor), and head(group g) * slot / sum of g's slots for a dynamic label.
    """
    h = np.full_like(s, LOW)
    h[0] = head[0]
    for i, c in enumerate(_anchors(meta)):
        h[int(c["label"])] = head[1 + i]
    slots = _slots(meta)
    gsum: Dict[int, np.ndarray] = {}
    for c in slots:
        gsum[c["group"]] = gsum.get(c["group"], 0) + p[int(c["index"])]
    for c in slots:
        h[int(c["label"])] = head[5 + int(c["group"])] * p[int(c["index"])] / np.maximum(gsum[c["group"]], eps)
    used = static_labels(meta) + dynamic_labels(meta)
    out = s.copy()
    out[used] = np.exp((1 - weight) * np.log(np.maximum(s[used], eps)) + weight * np.log(np.maximum(h[used], eps)))
    return out


def scale_dynamic(s: np.ndarray, dyn: Iterable[int], alpha: float) -> np.ndarray:
    """Multiply the dynamic-label scores by alpha (only the dynamic vs background/anchor decision can change)."""
    out = s.copy()
    dyn = list(dyn)
    out[dyn] = s[dyn] * np.float32(alpha)
    return out


def gated(base: np.ndarray, new: np.ndarray, dyn: Iterable[int], radius: int) -> np.ndarray:
    """Use `new` scores only within `radius` voxels (chessboard) of a voxel whose `base` argmax is dynamic."""
    dyn_mask = np.isin(np.argmax(base, axis=0), list(dyn))
    if radius > 0:
        dyn_mask = ndimage.binary_dilation(dyn_mask, structure=np.ones((3,) * dyn_mask.ndim, bool), iterations=radius)
    return np.where(dyn_mask[None], new, base)


def hysteresis(core: np.ndarray, relaxed: np.ndarray, dyn: Iterable[int]) -> np.ndarray:
    """Add relaxed voxels of dynamic label c on non-dynamic core voxels when 6-connected to a core component of c."""
    dyn = list(dyn)
    out = core.copy()
    grow_any = (relaxed != core) & ~np.isin(core, dyn) & np.isin(relaxed, dyn)
    if not grow_any.any():
        return out
    for lab in np.unique(relaxed[grow_any]):
        core_c = core == lab
        if not core_c.any():
            continue
        grow_c = grow_any & (relaxed == lab)
        comp, n = ndimage.label(core_c | grow_c)
        keep_lut = np.zeros(n + 1, dtype=bool)
        keep_lut[np.unique(comp[core_c])] = True
        keep_lut[0] = False
        out[grow_c & keep_lut[comp]] = lab
    return out
