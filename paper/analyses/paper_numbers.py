"""Every AnchorSlot-dependent number in the TMI manuscript, from one audit cohort directory.

usage: numbers.py <cohort_dir> <out_json>
Run on the old cohort first: it must reproduce the numbers printed in the paper; then on the new cohort.
Read-only on the cohort; writes only <out_json> and a CoNeMOS shim next to it.
"""
import json, math, sys
from pathlib import Path
import numpy as np

cohort, out_path = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
R = Path('/projects/weilab/liupeng/runs/Anchorslot')
sys.path.insert(0, str(cohort / 'source_dependencies'))
import report_refresh as rr          # noqa: E402
import robustness_stats as rs        # noqa: E402
import reliability_stats as rel      # noqa: E402

rd = lambda p: json.loads(Path(p).read_text())
report = rd(cohort / 'current_report.json')
crops = report['validation_crops']
CON = R / 'literature_baselines_f0_20261029/eval/conemos/official'
con = {c: v for c, v in rd(CON / 'metrics_16crop_extended.json').items() if c.startswith('crop')}
assert set(con) == set(crops)
cs = rr.combine(con)
ref = rd(CON / 'metrics_16crop.json')['scores']
assert all(math.isclose(cs[k], ref[k], abs_tol=1e-12) for k in ('semantic', 'instance', 'overall'))
shim = out_path.parent / (out_path.stem + '_conemos_shim')
shim.mkdir(exist_ok=True)
(shim / 'latest_official_metrics_extended.json').write_text(json.dumps(dict(
    con, overall_semantic_score=cs['semantic'], overall_instance_score=cs['instance'], overall_score=cs['overall'])))

KEYS = ['flat', 'cb_flat', 'binary', 'ternary', 'conemos', 'anchorslot']
METHODS = {k: (shim if k == 'conemos' else cohort / 'evaluation' / k) for k in KEYS}
data = {k: {c: v for c, v in rd(METHODS[k] / 'latest_official_metrics_extended.json').items() if c.startswith('crop')} for k in KEYS}
N = {}

# 1. scores, flagged pairs
N['scores'] = {k: {n: rr.combine(data[k])[n] for n in ('semantic', 'instance', 'overall')} for k in KEYS}
N['flagged'] = {k: sorted([c, l] for c in crops for l, v in data[k][c].items() if v['status'] != 'scored') for k in KEYS}

# 2. few-object sensitivity and count/volume error on pairs scored for every method (frozen helper)
rs.METHODS = METHODS
rs.OUT = out_path.parent / (out_path.stem + '_robustness.json')
rs.RNG_SEED = 20260930
rs.N_BOOT = 10000
rs.main.__globals__['print'] = lambda *a, **k: None
rs.main()
rob = rd(rs.OUT)
N['few_object_classes'] = rob['few_object_classes']
N['without_few'] = {k: rob['without_few_object_classes'][k]['overall'] for k in KEYS}
N['quant'] = rob['quantification']
N['bootstrap_overall_AS_minus'] = rob['bootstrap']['delta_overall_anchorslot_minus']

# 3. count error (signed medians, paired bootstrap vs each method)
rel.METHODS = METHODS
ce = rel.count_error_stats(rel.count_errors(data, crops), crops, np.random.default_rng(20260930))
N['count_error_point'] = ce['point']
N['count_error_delta'] = ce['delta_anchorslot_minus']

# 4. object-free pairs (from the cohort's own reliability file; CoNeMOS was never part of it)
of = rd(cohort / 'reliability_stats.json')['object_free']['stats']
N['object_free'] = {f: {'per_method': {m: s for m, s in of[f]['per_method'].items()},
                        'delta': of[f].get('delta_anchorslot_minus')} for f in of}

# 5. instance breakdown
def pool(d):
    cl = {}
    for c in crops:
        for name, r in d[c].items():
            if 'hausdorff_norm_sum' not in r:
                continue
            a = cl.setdefault(name, dict(tp=0, fp=0, fn=0, hs=0.0, nh=0))
            for k, s in (('tp', 'tp'), ('fp', 'fp'), ('fn', 'fn'), ('hs', 'hausdorff_norm_sum'), ('nh', 'n_hausdorff')):
                a[k] += r[s]
    for a in cl.values():
        f1 = 2 * a['tp'] / max(1, 2 * a['tp'] + a['fp'] + a['fn'])
        a['score'] = math.sqrt(f1 * a['hs'] / max(1, a['nh']))
    return cl
pooled = {k: pool(data[k]) for k in ('flat', 'cb_flat', 'conemos', 'anchorslot')}
for k in pooled:
    assert math.isclose(np.mean([c['score'] for c in pooled[k].values()]), N['scores'][k]['instance'], abs_tol=1e-9), k
names = sorted(pooled['anchorslot'], key=lambda n: (max(pooled[k][n]['tp'] + pooled[k][n]['fn'] for k in pooled), n))
N['instance_rows'] = {n: dict(objects=int(max(pooled[k][n]['tp'] + pooled[k][n]['fn'] for k in pooled)),
                              **{k: dict(tp=int(pooled[k][n]['tp']), fp=int(pooled[k][n]['fp']), score=pooled[k][n]['score']) for k in pooled})
                      for n in names}
N['instance_totals'] = {k: dict(tp=int(sum(v['tp'] for v in pooled[k].values())), fp=int(sum(v['fp'] for v in pooled[k].values()))) for k in pooled}
few = N['few_object_classes']
N['few_object_fp'] = {k: {n: int(pooled[k][n]['fp']) for n in few} | {'total': int(sum(pooled[k][n]['fp'] for n in few))} for k in pooled}
N['few_object_gt'] = int(sum(N['instance_rows'][n]['objects'] for n in few))
N['classes_AS_above_conemos'] = sorted(n for n in names if pooled['anchorslot'][n]['score'] > pooled['conemos'][n]['score'])
N['classes_AS_below_conemos'] = sorted(n for n in names if pooled['anchorslot'][n]['score'] < pooled['conemos'][n]['score'])

# 6. per-label Dice (pooled voxel counts), label roles
ANCH = ['ecs', 'pm', 'cyto', 'nucpl']
labels = list(report['models']['anchorslot']['dice'])
def dice_of(d):
    out = {}
    for l in labels:
        rows = [d[c][l] for c in crops if l in d[c]]
        tp, fp, fn = (sum(r[n] for r in rows) for n in ('tp', 'fp', 'fn'))
        out[l] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return out
dice = {k: dice_of(data[k]) for k in ('flat', 'cb_flat', 'conemos', 'anchorslot')}
for l in labels:
    assert math.isclose(dice['anchorslot'][l], report['models']['anchorslot']['dice'][l], abs_tol=1e-12)
dyn = [l for l in labels if l not in ANCH]
N['dice'] = dice
N['group_dice'] = {g: {k: 100 * float(np.mean([dice[k][l] for l in ls])) for k in dice}
                   for g, ls in (('anchors', ANCH), ('dynamic', dyn), ('all', labels))}
d_con = {l: 100 * (dice['anchorslot'][l] - dice['conemos'][l]) for l in labels}
d_flat = {l: 100 * (dice['anchorslot'][l] - dice['flat'][l]) for l in labels}
N['delta_vs_conemos'] = d_con
N['delta_vs_flat'] = d_flat
N['dynamic_without_perox_vs_conemos'] = float(np.mean([d_con[l] for l in dyn if not l.startswith('perox')]))
N['dynamic_wins_vs_flat'] = sum(d_flat[l] > 0 for l in dyn)
fam = ['ld', 'lyso', 'er', 'golgi', 'ne', 'perox', 'endo', 'eres', 'ves', 'mito']
N['mem_lum'] = {k: dict(mem=100 * float(np.mean([dice[k][f + '_mem'] for f in fam])), lum=100 * float(np.mean([dice[k][f + '_lum'] for f in fam])),
                        lum_gt_mem_all=all(dice[k][f + '_lum'] > dice[k][f + '_mem'] for f in fam)) for k in dice}

# 7. per-crop Dice of dynamic labels with > 5000 GT voxels: AnchorSlot vs CoNeMOS
big = []
for c in crops:
    for l in dyn:
        if l in data['anchorslot'][c] and l in data['conemos'][c]:
            a, b = data['anchorslot'][c][l], data['conemos'][c][l]
            if a['tp'] + a['fn'] > 5000:
                f = lambda r: 2 * r['tp'] / (2 * r['tp'] + r['fp'] + r['fn']) if 2 * r['tp'] + r['fp'] + r['fn'] else 1.0
                big.append((c, l, 100 * f(a), 100 * f(b)))
N['big_pairs'] = dict(n=len(big), AS_gain_gt20=sorted([x for x in big if x[2] - x[3] > 20], key=lambda x: x[3] - x[2]),
                      CoNeMOS_gain_gt20=sorted([x for x in big if x[3] - x[2] > 20], key=lambda x: x[2] - x[3]))

# 8. confusion of the 27 dynamic labels
ext = rd(cohort / 'confusion_extended.json')
raw = rd(cohort / 'raw_diagnostics.json')
N['confusion'] = {k: (ext['models'].get(k) or raw['models'][k]) for k in ('flat', 'cb_flat', 'binary', 'ternary', 'anchorslot')}
N['gt_background_percent'] = ext['gt_background_percent']
mkey = [k for k in ext if k.startswith('anchorslot_') and isinstance(ext[k], dict) and 'confusion' in ext[k]]
N['confusion_matrix_keys'] = mkey

# 9. size filter
N['size_filter_AS'] = {t: rd(cohort / f'uniform_filter_anchorslot_{t}.json') for t in ('nofilter', 'filtered', 'filtered_p10')}
N['size_filter_AS'] = {t: {k: v[k] for k in ('overall', 'semantic', 'instance', 'pathological')} for t, v in N['size_filter_AS'].items()}

out_path.write_text(json.dumps(N, indent=1, default=float) + '\n')
print('wrote', out_path)
