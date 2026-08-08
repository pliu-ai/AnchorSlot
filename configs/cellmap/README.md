# CellMap protocol snapshots

`paper_dataset200/` is the intended leakage-free paper configuration. Its
`imagesTs` cases remain outside the training set.

The source Dataset200 `dataset.json` declared 215 training cases, while the raw
snapshot and every saved fold contain 202 cases. The paper copy corrects
`numTraining` to 202 and `scripts/verify_protocol.py` enforces that count.

`reference_dataset202/` is retained only to reconstruct the completed July 2026
run. Dataset202's metadata explicitly says Dataset200 `imagesTr + imagesTs` and
`labelsTr + labelsTs` were merged into training. Results from that checkpoint on
Dataset200 `imagesTs` are therefore not held-out results.

These JSON files contain metadata, plans, fingerprints, and split identifiers;
they do not redistribute image data.

## Manifest-locked official evaluation

`scripts/cellmap_evaluation_pipeline.py` is the paper evaluation entry point.
It validates native instance IDs, hashes the exact annotated manifest and truth
validation report into a truth lock, exports predictions, runs the evaluator at
the required commit, and records code/evaluator provenance. The experiment
matrix is `evaluation_registry.tsv`.

The 32-nm holdout contains only native parent-instance annotations for `nuc`,
`mito`, and `perox`. Therefore the official semantic and geometric-overall
fields are not applicable for that cohort; compare its instance score. The
`common_parent_instance` protocol filters 4-nm evaluation to the same three
classes and is the valid cross-resolution comparison.

```bash
python scripts/submit_cellmap_evaluation_registry.py \
  --registry configs/cellmap/evaluation_registry.tsv

python scripts/cellmap_evaluation_pipeline.py summarize \
  --registry configs/cellmap/evaluation_registry.tsv \
  --output-dir /path/to/summary
```
