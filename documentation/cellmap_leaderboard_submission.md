# CellMap leaderboard submission

AnchorSlot predicts one mutually exclusive hard-label NIfTI volume with
background `0` and 31 atomic classes `1..31`. The CellMap leaderboard instead
expects overlapping per-class arrays in a Zarr-2 store. The adapter in
`nnunetv2/evaluation/cellmap_challenge.py` derives all requested atomic and
parent arrays without depending on the challenge Python package.

The adapter targets the evaluator behavior at challenge commit
`0300239cd0b4867d4bab008aa9e95161b2442d93`. Always download and retain the
manifest from the same evaluator revision used for a submission:

```bash
curl -L \
  https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/0300239cd0b4867d4bab008aa9e95161b2442d93/src/cellmap_segmentation_challenge/utils/test_crop_manifest.csv \
  -o /path/to/test_crop_manifest_0300239.csv
```

Install Zarr support. Either the Zarr 2 or 3 Python package may be installed;
the exporter always writes format 2 on disk:

```bash
pip install -e '.[cellmap]'
```

## Official test export

The prediction directory must contain one merged `*.nii.gz` file per crop at
its root. Files under `group_*` are conditioned intermediate outputs and are
intentionally ignored.

If predictions are split into resolution subdirectories, pass `--recursive`.
Directories named `group_*` and `split_labels` remain excluded.

```bash
anchorslot_export_cellmap_submission \
  --predictions /path/to/official_test_predictions \
  --manifest /path/to/test_crop_manifest_0300239.csv \
  --dry-run \
  --report /path/to/input_validation_report.json

anchorslot_export_cellmap_submission \
  --predictions /path/to/official_test_predictions \
  --manifest /path/to/test_crop_manifest_0300239.csv \
  --output /path/to/submission.zarr \
  --report /path/to/export_report.json

anchorslot_validate_cellmap_submission \
  --submission /path/to/submission.zarr \
  --manifest /path/to/test_crop_manifest_0300239.csv \
  --report /path/to/validation_report.json \
  --zip
```

`--zip` is intentionally attached to validation: an archive is produced only
after every strict check passes.

For the full 16-crop export on Slurm, use the provided long-running job:

```bash
REPO_ROOT=/path/to/AnchorSlot \
PREDICTIONS=/path/to/official_test_predictions \
MANIFEST=/path/to/test_crop_manifest_0300239.csv \
OUTPUT=/path/to/submission.zarr \
RECURSIVE_PREDICTIONS=1 \
PYTHON=/path/to/python \
sbatch scripts/slurm/export_cellmap_submission.sbatch
```

The exporter:

- rejects missing crop predictions and labels outside `0..31`;
- requires the prediction shape to equal the manifest shape exactly;
- writes every requested crop/class pair, including all-zero masks;
- derives hard parent masks as logical unions of atomic children;
- writes `voxel_size`, `translation`, and `shape` from the manifest rather than
  trusting NIfTI origin metadata;
- records the manifest SHA-256 and source prediction directory in root attrs;
- filters output to exactly the manifest pairs, so unrelated arrays cannot
  enter the submitted archive.

The validator checks the Zarr-2 marker, exact manifest coverage, absence of
extra arrays, binary values, spatial metadata, manifest hash, and all parent
unions for crops where the manifest includes the corresponding children.

## Validation predictions

Legacy `imagesTs` predictions do not use the current leaderboard crop IDs. A
derived all-class manifest can be used to exercise the identical conversion
and validation code without claiming an official test score:

```bash
anchorslot_build_cellmap_validation_manifest \
  --predictions /path/to/validation_predictions \
  --output /path/to/validation_manifest.csv

anchorslot_export_cellmap_submission \
  --predictions /path/to/validation_predictions \
  --manifest /path/to/validation_manifest.csv \
  --output /path/to/validation_submission.zarr

anchorslot_validate_cellmap_submission \
  --submission /path/to/validation_submission.zarr \
  --manifest /path/to/validation_manifest.csv
```

This verifies the submission representation only. It does not turn legacy
validation crops into an official leaderboard evaluation and does not provide
access to hidden test ground truth.

When local per-class validation annotations are available, first filter the
all-class manifest so absent annotation files are treated as unannotated rather
than empty masks:

```bash
anchorslot_build_cellmap_annotated_validation_manifest \
  --manifest /path/to/validation_manifest.csv \
  --labels-root /path/to/labelsTr_split \
  --output /path/to/validation_manifest_annotated.csv
```

The reproducible long-running workflow in
`scripts/slurm/evaluate_cellmap_validation_latest.sbatch` exports prediction
and truth stores, validates both, and evaluates them with a pinned official
challenge checkout. Set `ANCHORSLOT_ROOT`, `PREDICTIONS`,
`GROUND_TRUTH_LABELS_ROOT`, `MANIFEST`, `OUTPUT_DIR`, `CSC_REPO`, and `PYTHON` when
submitting it.

Ground truth must use the native per-class tree, not a merged 0..31 NIfTI:

```bash
anchorslot_export_cellmap_submission \
  --truth-labels-root /path/to/labelsTr_split \
  --manifest /path/to/validation_manifest_annotated.csv \
  --output /path/to/ground_truth.zarr

anchorslot_validate_cellmap_submission \
  --submission /path/to/ground_truth.zarr \
  --manifest /path/to/validation_manifest_annotated.csv \
  --role ground_truth
```

This distinction is essential. Prediction arrays are binary and the official
instance scorer connected-components them. Instance-scored truth arrays must
retain their native integer object IDs. Exporting truth through
`--predictions` collapses multiple objects to ID 1 and invalidates the instance
metric. Ground-truth validation now rejects a binary array when it contains
multiple connected objects.

For a quick integration test on one crop, add `--crop 174` when building the
validation manifest. The export will then contain only `crop174` and its 48
arrays.
