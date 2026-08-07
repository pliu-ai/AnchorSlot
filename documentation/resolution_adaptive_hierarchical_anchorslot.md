# Resolution-Adaptive Hierarchical Parallel AnchorSlot

This is the multi-resolution method variant intended for the TMI study. It is
separate from the HPA baseline so the original implementation and checkpoints
remain reproducible.

## Model

One encoder/decoder is shared across resolutions. Every batch carries its
physical voxel size `v`; the network encodes `log2(v / 4 nm)` and uses the
embedding in two places:

1. feature-wise modulation at every decoder stage;
2. batch-conditioned group/anchor-slot codes.

The decoder emits five aligned outputs:

- the normalized 32-way atomic distribution;
- the original HPA coarse taxonomy;
- the 12 x 2 parallel anchor slots;
- 17 sigmoid outputs matching the official overlapping CellMap parent DAG;
- one boundary and one nearest-neighbor affinity per spatial axis.

The direct parent probabilities are constrained to agree with the sum of their
atomic descendants. The separation head is class-agnostic to keep full-resolution
memory manageable. With merged labels it receives a conservative organelle-family
edge prior. Native instance IDs can be passed to the loss when an instance-sidecar
dataloader is available; experiments must distinguish this true supervision from
the weak family auxiliary.

## Annotation-aware mixed-resolution training

The trainer can alternate two native nnU-Net preprocessed streams without
resampling every case to one common physical scale. The paper configuration is:

- primary: Dataset200, `3d_lowres_large_patch`, approximately 4 nm;
- auxiliary: Dataset201, `3d_fullres`, 32 nm;
- auxiliary sampling probability: 0.5;
- shared model and optimizer; separate augmentation streams;
- validation alternates both streams and logs Dice by resolution.

`NNUNET_RAHPA_NATIVE_LABELS_ROOT` points to the native per-class annotations.
For every case, atomic and parent losses are activated only for annotation files
that actually exist. Inactive atomic channels are removed from within-crop
softmax competition, so an absent organelle annotation is not treated as a
negative label. Background remains active as the negative class for the classes
that are annotated in that crop.

Some 32-nm merged segmentations encode a parent-only annotation with a
representative descendant ID. In that situation the representative ID is masked
from atomic supervision, the native parent filename activates only the correct
parent loss, and hierarchy consistency transfers that supervision to the sum of
the parent's descendants without asserting a false membrane/lumen label.

## Training

```bash
sbatch scripts/slurm/train_resolution_adaptive_hierarchical_anchorslot.sbatch
```

Hyperparameters are in
`configs/resolution_adaptive_hierarchical_anchorslot.env`. Useful ablations are
obtained by setting the parent, hierarchy, boundary, affinity, or auxiliary-data
weights/probability to zero. Do not change more than one factor per ablation.

## Inference

The structured predictor accepts the physical voxel size:

```bash
python scripts/predict_structured_conditional.py \
  -i /path/to/images \
  -o /path/to/predictions_32nm \
  -m /path/to/model \
  --group_id all \
  --output_mode original \
  --voxel_size_nm 32
```

Use `--voxel_size_nm 4` for the high-resolution stream and an anisotropic vector
such as `--voxel_size_nm 8,8,32` when appropriate. The value is saved in the
prediction metadata.

## Required TMI experiment matrix

At minimum, run three seeds for:

1. high-only HPA;
2. low-only HPA;
3. mixed data without scale conditioning;
4. mixed data plus scale conditioning;
5. full model without direct parent/hierarchy losses;
6. full model without separation losses;
7. the complete model.

Report semantic and instance scores at 4, 16, and 32 nm, per-organelle results,
subject-level bootstrap confidence intervals, parameters, peak memory, and
sliding-window throughput. The official evaluator commit and manifest hash must
be recorded for every table.
