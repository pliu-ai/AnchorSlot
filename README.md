# AnchorSlot

Official implementation snapshot for **Hierarchical Parallel AnchorSlot**, built
on nnU-Net v2 for structured 3D connectomics segmentation.

The method predicts all 12 CellMap organelle groups in one encoder/decoder pass.
A 20-way coarse taxonomy selects background, fixed structures, or an organelle
group; two reusable anchor slots then resolve each dynamic membrane/lumen pair.
The two terms are composed into a normalized 32-label distribution.

## What this repository contains

- the complete nnU-Net v2 framework needed by the trainer;
- the Hierarchical Parallel AnchorSlot network, hierarchy mapping, joint loss,
  trainer, and single-pass predictor;
- exact Dataset200 and historical Dataset202 metadata/splits;
- leakage-guarded training, prediction, and evaluation commands;
- a manifest-pinned CellMap leaderboard Zarr-2 exporter and validator;
- the filtered metric implementation used by the experiment notes;
- lightweight unit tests that do not require CellMap data or a GPU.

The main implementation is under
`nnunetv2/training/nnUNetTrainer/variants/structured_conditional/`:

- `hierarchical_parallel_mapping.py`
- `network_hierarchical_parallel_anchorslot.py`
- `structured_loss_hierarchical_parallel_anchorslot.py`
- `trainer_hierarchical_parallel_anchorslot.py`
- `inference_hierarchical_parallel_anchorslot.py`

The multi-resolution upgrade is implemented separately so baseline checkpoints
remain compatible:

- `network_resolution_adaptive_hierarchical_anchorslot.py`
- `resolution_adaptive_mapping.py`
- `structured_loss_resolution_adaptive_hierarchical_anchorslot.py`
- `trainer_resolution_adaptive_hierarchical_anchorslot.py`

It trains one shared model on real 4 nm and 32 nm batches, conditions decoder
features and slot codes on physical voxel size, directly supervises all 17
official parent nodes, and adds boundary/affinity auxiliaries. See
`documentation/resolution_adaptive_hierarchical_anchorslot.md`.

## Important evaluation-protocol warning

The archived `0.5167` foreground Dice result is **not a valid held-out paper
result**. Its checkpoint was trained on Dataset202, whose metadata states that
Dataset200 `imagesTs/labelsTs` were merged into its training set, and was then
evaluated on those same 17 `imagesTs` cases. It is retained in
`results/reference/` only to verify the historical pipeline.

Paper experiments must use Dataset200 (the default in the release scripts) or a
new subject-disjoint split that never places the evaluation cases in training.
`scripts/train_anchorslot.sh` refuses Dataset202 unless the user explicitly sets
`ANCHORSLOT_ALLOW_LEAKY_REFERENCE=1`.

## Installation

Python 3.12 and PyTorch 2.5.1 reproduce the completed reference run. Create the
environment and install this repository in editable mode:

```bash
conda env create -f environment.yml
conda activate anchorslot
pip install -e .
```

Set the standard nnU-Net paths:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

The CellMap data must be prepared as
`Dataset200_Dataset101_CellMap_model_high_res`. Dataset metadata, plans, and the
five fixed folds used locally are preserved under
`configs/cellmap/paper_dataset200/`.

## Training

The paper-safe default is Dataset200, configuration `3d_lowres_large_patch`,
fold 0, 3,000 epochs, batch size 1, and the exact hierarchical loss weights used
by the completed run:

```bash
bash scripts/train_anchorslot.sh
```

Arguments may override dataset ID, configuration, and fold:

```bash
bash scripts/train_anchorslot.sh 200 3d_lowres_large_patch 0
```

For Slurm, edit resource requests if needed and submit:

```bash
sbatch scripts/slurm/train_anchorslot.sbatch
```

For joint 4/32-nm training with Dataset200 as the primary stream and Dataset201
as the auxiliary stream:

```bash
sbatch scripts/slurm/train_resolution_adaptive_hierarchical_anchorslot.sbatch
```

## Prediction

```bash
MODEL_FOLDER=/path/to/trained/model \
INPUT_DIR="$nnUNet_raw/Dataset200_Dataset101_CellMap_model_high_res/imagesTs" \
OUTPUT_DIR=/path/to/predictions \
bash scripts/predict_anchorslot.sh
```

Hierarchical Parallel AnchorSlot directly outputs original labels 0–31 and uses
one network pass per sliding-window tile. The legacy `--group_id all` argument
is accepted for compatibility but does not trigger 12 conditioned passes.

## Evaluation

First create the per-case present-label JSON from the ground truth if it is not
already available. Then run:

```bash
PREDICTIONS=/path/to/predictions \
GROUND_TRUTH="$nnUNet_raw/Dataset200_Dataset101_CellMap_model_high_res/labelsTs" \
UNIQUE_LABELS_JSON=/path/to/labelsTs_unique_labels_sum.json \
bash scripts/evaluate_anchorslot.sh
```

This writes both nnU-Net's `summary.json` and the historical
present-label-filtered `summary_filtered.json`. For a TMI paper, report the
metric definition explicitly and add subject-level confidence intervals; do not
use the filtered mean as the only outcome.

### Official CellMap leaderboard format

The leaderboard does not consume the historical `summary_filtered.json` or a
single multiclass NIfTI. Use the independent manifest-aware exporter to derive
the 48 overlapping atomic/parent arrays and validate their Zarr-2 geometry:

```bash
pip install -e '.[cellmap]'

anchorslot_export_cellmap_submission \
  --predictions /path/to/official_test_predictions \
  --manifest /path/to/pinned/test_crop_manifest.csv \
  --output /path/to/submission.zarr

anchorslot_validate_cellmap_submission \
  --submission /path/to/submission.zarr \
  --manifest /path/to/pinned/test_crop_manifest.csv
```

When evaluating local validation data, native instance-labelled truth must be
exported with `--truth-labels-root` and validated with `--role ground_truth`.
Do not export a merged truth NIfTI through the prediction adapter.

See [the leaderboard submission guide](documentation/cellmap_leaderboard_submission.md)
for pinned-manifest, validation-prediction, and packaging commands.

## Tests

The smoke suite is CPU-only:

```bash
pytest -q nnunetv2/tests/test_hierarchical_parallel_anchorslot.py
pytest -q nnunetv2/tests/test_cellmap_challenge_submission.py
python scripts/verify_protocol.py --config-root configs/cellmap
```

## Provenance and licensing

This clean snapshot was extracted from the development fork at commit
`44152023d1edb51deea9d5be49ea09d0c0d07ba2` and overlaid with the exact
uncommitted Hierarchical Parallel AnchorSlot files used for the July 2026 run.
See `SOURCE_PROVENANCE.md` for details.

nnU-Net and this derivative repository are distributed under Apache-2.0. Keep
`LICENSE` and `THIRD_PARTY_NOTICES.md` with redistributed copies. Cite nnU-Net
in addition to the AnchorSlot paper.

## Status

This repository is a reproducible development release. Before the TMI artifact
is tagged, the remaining requirements are leakage-free retraining, independent
seed runs, final weight publication, and replacement of the placeholder paper
metadata in `CITATION.cff`.
