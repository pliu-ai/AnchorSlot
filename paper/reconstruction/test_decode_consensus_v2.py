#!/usr/bin/env python3
"""Unit tests for Step 1 decode_v2.py mathematical guarantees."""
from __future__ import annotations

import numpy as np
import decode_consensus_v2 as dv2


def _dummy_meta():
    meta = [
        {"index": 0, "group": -1, "kind": "background_mean", "label": 0},
        {"index": 1, "group": -1, "kind": "anchor_mean", "label": 1},
        {"index": 2, "group": -1, "kind": "anchor_mean", "label": 2},
        {"index": 3, "group": -1, "kind": "anchor_mean", "label": 3},
        {"index": 4, "group": -1, "kind": "anchor_mean", "label": 27},
    ]
    groups = [
        (0, [4, 5, 6]), (1, [7, 8]), (2, [9, 10]), (3, [11, 12]), (4, [13, 14]),
        (5, [15, 16]), (6, [19, 20]), (7, [25, 26]), (8, [21, 22]), (9, [23, 24]),
        (10, [28, 29]), (11, [30, 31]), (12, [17, 18]),
    ]
    for g, labs in groups:
        for lab in labs:
            meta.append({"index": len(meta), "group": g, "kind": "slot", "label": lab})
        meta.append({"index": len(meta), "group": g, "kind": "other", "label": -1})
    n_c = len(meta)
    for k in range(18):
        meta.append({"index": n_c + k, "group": -1, "kind": "group_head", "label": k})
    return meta


def test_twostage_eliminates_vote_splitting():
    meta = _dummy_meta()
    compact, head_idx, anchors, slots, others, g2s = dv2.parse_meta(meta)
    p = np.full((len(compact), 2, 2, 2), 1e-5, dtype=np.float32)
    head = np.full((18, 2, 2, 2), 1e-5, dtype=np.float32)
    # Voxel (0,0,0): cyto=0.42, ER total=0.58 split as er_mem=0.32, er_lum=0.26
    p[3, 0, 0, 0] = 0.42  # cyto
    er_mem_idx, er_lum_idx = g2s[12][0][1], g2s[12][1][1]
    p[er_mem_idx, 0, 0, 0] = 0.32
    p[er_lum_idx, 0, 0, 0] = 0.26
    head[3, 0, 0, 0] = 0.42     # cyto
    head[5 + 12, 0, 0, 0] = 0.58  # ER group

    gmix = dv2.build_gmix_scores(p, head, anchors, slots, weight=0.5)
    ts = dv2.build_twostage_scores(p, head, anchors, g2s, weight=0.5)
    # Flat Gmix suffers vote-splitting and wrongly chooses cyto (3)
    assert int(np.argmax(gmix[:, 0, 0, 0])) == 3
    # Two-Stage chooses ER group (0.58 > 0.42) and within ER picks er_mem (17)
    assert int(np.argmax(ts[:, 0, 0, 0])) == 17


def test_twostage_exact_equivalence_to_explicit_two_stage():
    rng = np.random.default_rng(42)
    meta = _dummy_meta()
    compact, head_idx, anchors, slots, others, g2s = dv2.parse_meta(meta)
    p = rng.uniform(0.01, 1.0, size=(len(compact), 4, 4, 4)).astype(np.float32)
    head = rng.uniform(0.01, 1.0, size=(18, 4, 4, 4)).astype(np.float32)
    head /= head.sum(axis=0, keepdims=True)
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        S32 = dv2.build_twostage_scores(p, head, anchors, g2s, weight=w)
        pred = np.argmax(S32, axis=0)
        # Explicit reference two-stage
        Q_slot = np.zeros((18, 4, 4, 4), dtype=np.float32)
        Q_slot[0] = p[0]
        for i, c in enumerate(anchors):
            Q_slot[1 + i] = p[int(c["index"])]
        for g in range(13):
            Q_slot[5 + g] = sum(p[idx] for _, idx in g2s[g])
        Q_f = np.power(Q_slot, 1.0 - w) * np.power(head, w)
        coarse = np.argmax(Q_f, axis=0)
        ref = np.zeros((4, 4, 4), dtype=np.int64)
        for i, c in enumerate(anchors):
            ref[coarse == (1 + i)] = int(c["label"])
        for g in range(13):
            m = coarse == (5 + g)
            if m.any():
                arr = np.stack([p[idx] for _, idx in g2s[g]], axis=0)
                labs = np.array([lab for lab, _ in g2s[g]])
                ref[m] = labs[np.argmax(arr, axis=0)][m]
        assert np.array_equal(pred, ref), f"Mismatch at w={w}"


def test_hysteresis_zero_new_components():
    meta = _dummy_meta()
    _, _, _, slots, _, g2s = dv2.parse_meta(meta)
    dyn = [int(c["label"]) for c in slots]
    core = np.full((5, 5, 5), 3, dtype=np.uint8)  # cyto
    core[2, 2, 2] = 5  # mito_lum core
    relaxed = core.copy()
    relaxed[2, 2, 1] = 4  # mito_mem 6-adjacent to mito_lum core -> SHOULD GROW!
    relaxed[0, 0, 0] = 4  # disconnected mito_mem speck -> MUST NOT GROW!
    out = dv2.hysteresis_grow_groups(core, relaxed, g2s, dyn)
    assert out[2, 2, 1] == 4
    assert out[0, 0, 0] == 3


def test_veto_removes_only_high_conflict_small_components():
    meta = _dummy_meta()
    compact, _, anchors, slots, others, g2s = dv2.parse_meta(meta)
    p = np.full((len(compact), 6, 6, 6), 1e-4, dtype=np.float32)
    head = np.full((18, 6, 6, 6), 1e-4, dtype=np.float32)
    p[3] = 0.9; head[3] = 0.9
    seg = np.full((6, 6, 6), 3, dtype=np.uint8)
    # Component 1 (TP): clean lysosome (group 4, label 14) with low own_other=0.01 and head==slot agreement
    g_lyso = 4
    lyso_idx = g2s[g_lyso][1][1]
    seg[1, 1, 1] = 14
    p[3, 1, 1, 1] = 0.05; p[lyso_idx, 1, 1, 1] = 0.94; p[others[g_lyso]["index"], 1, 1, 1] = 0.01
    head[3, 1, 1, 1] = 0.05; head[5 + g_lyso, 1, 1, 1] = 0.94
    # Component 2 (FP): high own_other=0.35 + head disagrees
    seg[4, 4, 4] = 14
    p[3, 4, 4, 4] = 0.10; p[lyso_idx, 4, 4, 4] = 0.55; p[others[g_lyso]["index"], 4, 4, 4] = 0.35
    head[3, 4, 4, 4] = 0.60; head[5 + g_lyso, 4, 4, 4] = 0.30

    vm = dv2.compute_veto_mask_4nm(p, head, anchors, others, g2s, seg)
    assert not vm[1, 1, 1], "Clean TP component must NOT be vetoed"
    assert vm[4, 4, 4], "High-conflict FP component MUST be vetoed"


def test_la_prior_adjustment_boosts_rare_group():
    rng = np.random.default_rng(7)
    head = np.full((18, 2, 2, 2), 1.0 / 18.0, dtype=np.float32)
    priors = np.array([0.01, 0.2, 0.05, 0.5, 0.1] + [0.05] * 12 + [0.0005], dtype=np.float32)
    priors /= priors.sum()
    adj = dv2.adjust_head_with_priors(head, priors, tau=0.25)
    assert np.allclose(adj.sum(axis=0), 1.0, atol=1e-5)
    assert adj[17, 0, 0, 0] > adj[3, 0, 0, 0]


if __name__ == "__main__":
    test_twostage_eliminates_vote_splitting()
    test_twostage_exact_equivalence_to_explicit_two_stage()
    test_hysteresis_zero_new_components()
    test_veto_removes_only_high_conflict_small_components()
    test_la_prior_adjustment_boosts_rare_group()
    print("ALL_DECODE_V2_TESTS_PASSED")
