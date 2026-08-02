# Hierarchical Parallel AnchorSlot

`nnUNetTrainerHierarchicalParallelAnchorSlot` is the single-pass, coarse-to-fine
AnchorSlot variant for the 32-label CellMap setup.

The model uses one shared nnUNet encoder/decoder. At every supervised decoder
scale it predicts:

- 20 coarse categories: background, seven fixed labels, and 12 dynamic groups;
- two reusable anchor slots for every dynamic group, evaluated in parallel;
- 32 original-label log-probabilities composed from the coarse and slot terms.

For a dynamic label belonging to group `g` and slot `s`, the model uses
`log p(g | x) + log p(s | g, x)`. This guarantees a normalized 32-class output
without a heuristic merge and replaces the 12 conditioned inference passes with
one pass.

## Training

Use the trainer class with the normal nnUNet training entrypoint:

```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerHierarchicalParallelAnchorSlot
```

Useful optional environment variables:

```bash
export NNUNET_HPA_CODE_DIM=64
export NNUNET_HPA_LAMBDA_SEMANTIC_CE=1.0
export NNUNET_HPA_LAMBDA_SEMANTIC_DICE=1.0
export NNUNET_HPA_LAMBDA_COARSE_CE=0.5
export NNUNET_HPA_LAMBDA_SLOT_CE=0.5
```

The regular nnUNet predictor receives the 32-class semantic output. For analysis,
`predict_hierarchical_parallel(..., return_hierarchy=True)` also returns coarse
and per-group slot logits. `scripts/predict_structured_conditional.py` also
detects this trainer and automatically skips its legacy 12-group sweep; use
`--output_mode original` (the `--group_id` value is ignored for this model).
