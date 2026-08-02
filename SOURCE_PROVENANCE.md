# Source provenance

## Development source

- Development repository: `git@github.com:luckieucas/nnUNet-connectomics.git`
- Base commit: `44152023d1edb51deea9d5be49ea09d0c0d07ba2`
- Base commit date: 2026-07-22
- Base commit subject: `Add AnchorSlot structured conditional segmentation variants`
- Release snapshot assembled: 2026-08-02

The development repository contained a very large Git object database, so this
paper repository was created as a clean working-tree snapshot rather than by
copying that history. All files tracked at the base commit were exported with
`git archive`. The following working-tree files were then overlaid verbatim:

- `nnunetv2/training/nnUNetTrainer/variants/structured_conditional/__init__.py`
- `nnunetv2/training/nnUNetTrainer/variants/structured_conditional/hierarchical_parallel_mapping.py`
- `nnunetv2/training/nnUNetTrainer/variants/structured_conditional/inference_hierarchical_parallel_anchorslot.py`
- `nnunetv2/training/nnUNetTrainer/variants/structured_conditional/network_hierarchical_parallel_anchorslot.py`
- `nnunetv2/training/nnUNetTrainer/variants/structured_conditional/structured_loss_hierarchical_parallel_anchorslot.py`
- `nnunetv2/training/nnUNetTrainer/variants/structured_conditional/trainer_hierarchical_parallel_anchorslot.py`
- `nnunetv2/tests/test_hierarchical_parallel_anchorslot.py`
- `scripts/predict_structured_conditional.py`
- `documentation/hierarchical_parallel_anchorslot.md`

The completed reference checkpoint reports PyTorch 2.5.1, cuDNN 9.1.0, an
NVIDIA L40S, code dimension 64, learning rate 0.001, weight decay 3e-5, batch
size 1, 250 training iterations/epoch, 50 validation iterations/epoch, and
3,000 epochs.

The original Dataset200 metadata declared `numTraining=215`, but the available
raw dataset and each saved fold contain 202 unique training cases. The
paper-protocol copy corrects this field to 202; the unmodified source metadata
remains outside this repository and the discrepancy is checked by
`scripts/verify_protocol.py`.

## Upstream

The base framework derives from nnU-Net v2 by MIC-DKFZ. The upstream project is
available at <https://github.com/MIC-DKFZ/nnUNet> under Apache-2.0.

## Result provenance

`results/reference/summary_filtered.json` is copied from:

`runs/rahseg/hierarchical_parallel_anchorslot_exp01/predict_imagesTs_best_gall_original/20260730/predictions/summary_filtered.json`

It contains 17 cases, 225 present foreground label-cases, foreground Dice
0.5167002009763103, foreground IoU 0.4040599380946998, and 31-class macro Dice
0.47125917003260115. Because the model used Dataset202, this result has training
to evaluation leakage and must not be used as the paper's held-out claim.
