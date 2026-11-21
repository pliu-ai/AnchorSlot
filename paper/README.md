# Paper scripts: AnchorSlot (TMI submission)

Scripts behind the figures, tables, and robustness analyses of
"AnchorSlot: Role-Structured Supervision for Pan-Organelle Segmentation in Volume EM".
The model code is in `nnunetv2/` of this repository; this folder only reproduces the paper's figures and analyses
from trained-model predictions.

All scoring uses the official CellMap evaluator
([janelia-cellmap/cellmap-segmentation-challenge](https://github.com/janelia-cellmap/cellmap-segmentation-challenge),
commit `0300239`) on the 17 internal validation crops
(15, 20, 31, 50, 118, 121, 125, 126, 164, 174, 190, 219, 235, 236, 266, 268, 342).

## Paper components and their code

| Paper component | Code |
|---|---|
| AnchorSlot (nnU-Net / PlainConvUNet) | `nnunetv2/training/nnUNetTrainer/variants/structured_conditional/trainer_structured_conditional_er_dynamic.py`, `structured_loss_er_dynamic.py`, `dataloader_structured_conditional_er_dynamic.py`, `label_mapping_er_dynamic.py`, `network_structured_conditional.py` |
| AnchorSlot (MedNeXt) / flat MedNeXt | `trainer_structured_conditional_er_dynamic_mednext.py`, `trainer_mednext_32out_no_cond.py` |
| Leaderboard model (extra reweighting stage) | `trainer_structured_conditional_er_dynamic_ddw.py`, `structured_loss_er_dynamic_ddw.py`, `nnunetv2/training/loss/dynamic_distribution_weighting.py` |
| Label-organization and earlier loss variants | `trainer_structured_conditional_no_slot3*.py` |
| Synapse CT | `trainer_synapse_reconstruction.py` (`nnUNetTrainerSynapseReconstructionScreen`), `synapse_reconstruction.py`; run drivers `scripts/run_synapse_fold0_followup.py` (arm `lr003`), `scripts/continue_synapse_masked.py`, `scripts/snapshot_synapse_checkpoints.py` |
| Binary / three-class conditional baselines | `nnunetv2/training/nnUNetTrainer/variants/partial_label/nnUNetTrainerConditionalFiLM.py`, `nnUNetTrainerConditionalFiLMTernary.py` |
| Class-balanced flat baseline | `nnunetv2/training/nnUNetTrainer/nnUNetTrainerRalph.py` |
| Inference with query sweep and reconstruction | `scripts/predict_structured_conditional.py --group_id all`, `inference_structured_conditional_er_dynamic.py` |
| Encoder-cached inference (Section V-G) | `scripts/predict_structured_conditional.py --group_id all --cache_encoder`; tests in `nnunetv2/tests/test_cached_encoder_inference.py` |
| Computational cost (Table VIII) | `paper/efficiency/`, `paper/figures/make_cost_table.py` |
| Official-protocol evaluation | `scripts/cellmap_evaluation_pipeline.py`, `nnunetv2/evaluation/cellmap_challenge.py` |

## Scripts in this folder

`figures/` renders Figs. 1 and 3–6 and generates Tables IV and the robustness and confusion tables.
`analyses/` computes:
- the paired crop bootstrap, few-object sensitivity, and organelle count and volume errors (`robustness_stats.py`);
- the confusion analysis for all methods (`confusion_extended.py`);
- the uniform size-filter check (`uniform_filter.py`).

`efficiency/` measures the cost table on one GPU:
- `efficiency_metrics.py`: parameters, FLOPs per $160^3$ tile (`torch.utils.flop_counter`) and peak training memory
  of each trainer's `train_step` on synthetic batches;
- `benchmark_inference.py`: network and total time per validation volume for flat, class-balanced flat, per-query
  AnchorSlot and encoder-cached AnchorSlot, with a voxel-wise check that the cached and per-query label maps agree;
- `single_group_timing.py`: time of one query group per volume;
- `methods.py`: the checkpoints and training environments.

`figures/make_cost_table.py` writes the table from their JSON outputs.

With `--cache_encoder`, each mirrored tile is encoded once and decoded for all 13 query groups, since FiLM conditions
only the decoder. The logits are identical to the default per-query passes (tests and a GPU check on all 17
validation volumes) and inference is about 1.5 times faster. The price is one fp32 logit accumulator per group on the
GPU (about 35 GB for a 400^3 volume); groups are split into chunks automatically when they do not fit.

`figures/fig02_method.tex` is the standalone TikZ source of Fig. 2; compile it with XeLaTeX.

These scripts were written for our cluster layout. Paths to predictions, the locked ground-truth store, and the
evaluator checkout are absolute (`/projects/weilab/...`) and must be adapted. `analyses/confusion_extended.py` and `figures/make_label_figures.py` read the
validation label maps and EM images from the environment variables `CELLMAP_LABELS_TS` and `CELLMAP_IMAGES_TS`. The figure scripts also expect to live
in `scripts/` of the manuscript folder, which contains `figures/` and `tables/`.
