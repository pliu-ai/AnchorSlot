# Paper Scripts: AnchorSlot (IEEE TMI Submission)

Scripts reproducing the figures, tables, reconstruction variants, and statistical analyses of
**"AnchorSlot: Role-Structured Supervision for Pan-Organelle Segmentation in Volume EM"**.

All scoring uses the official CellMap evaluator
([`janelia-cellmap/cellmap-segmentation-challenge`](https://github.com/janelia-cellmap/cellmap-segmentation-challenge),
commit `0300239`) on the **16-crop ROI-audited primary validation benchmark** (and the 17-crop split retained for sensitivity checks; see [`configs/cellmap/source_roi_audit_16crop.json`](../configs/cellmap/source_roi_audit_16crop.json)).

## Paper Components and Code Mapping

| Paper Component | Code Location |
|---|---|
| AnchorSlot (PlainConvUNet, no group head) | `nnunetv2/training/nnUNetTrainer/variants/structured_conditional/trainer_structured_conditional_er_dynamic.py`, `structured_loss_er_dynamic.py`, `dataloader_structured_conditional_er_dynamic.py`, `label_mapping_er_dynamic.py`, `network_structured_conditional.py` |
| AnchorSlot + Group Head ($\lambda_{\text{gh}}=0.3$) | `trainer_structured_conditional_er_dynamic_group_head.py`, `scripts/predict_dump_group_head.py`, `scripts/merge_group_head.py` |
| Non-query-as-background factorial arm (`other -> bg`) | `trainer_structured_conditional_er_dynamic_group_head_nqbg.py`, `paper/reconstruction/decode_factorial_controls.py` |
| Controlled training ablations (`Pos100`, `Unmasked`, `Unweighted`) | `trainer_structured_conditional_er_dynamic_ablations.py`, `nnunetv2/tests/test_anchorslot_training_ablations.py` |
| Two-Stage & Consensus-Veto reconstruction (`TS_w0.5`, `TS_w0.5_g0.75_veto`) | `paper/reconstruction/decode_consensus_v2.py`, `paper/reconstruction/evaluate_cv5_consensus.py`, `paper/reconstruction/test_decode_consensus_v2.py` |
| AnchorSlot (MedNeXt-B) / Flat MedNeXt-B | `trainer_structured_conditional_er_dynamic_mednext.py`, `trainer_mednext_32out_no_cond.py` |
| Leaderboard model (extra DDW reweighting stage) | `trainer_structured_conditional_er_dynamic_ddw.py`, `structured_loss_er_dynamic_ddw.py`, `nnunetv2/training/loss/dynamic_distribution_weighting.py` |
| Flat nnU-Net / Class-Balanced Flat / Flat + Group Head | `nnunetv2/training/nnUNetTrainer/nnUNetTrainerRalph.py`, `nnUNetTrainerRalphGroupHead.py` |
| Binary / Ternary FiLM baselines | `nnunetv2/training/nnUNetTrainer/variants/partial_label/nnUNetTrainerConditionalFiLM.py`, `nnUNetTrainerConditionalFiLMTernary.py` |
| Adapted literature baselines (CoNeMOS, UniSeg, MISO-style) | `nnunetv2/training/nnUNetTrainer/nnUNetTrainerLiterature.py`, `literature_baselines/`, `scripts/predict_literature_baselines.py`, `nnunetv2/tests/test_literature_baselines.py` |
| Adapted DoDNet, Blob Loss, HSSN, SALT baselines | `nnunetv2/training/nnUNetTrainer/variants/comparison_baselines/`, `scripts/run_comparison_baseline.py`, `nnunetv2/tests/test_comparison_*.py` |
| External transfer on Synapse Abdominal CT | `trainer_synapse_reconstruction.py`, `synapse_reconstruction.py`, `scripts/run_synapse_fold0_followup.py` |
| Encoder-cached inference ($1.47\times$ speedup) | `scripts/predict_structured_conditional.py --group_id all --cache_encoder`, `nnunetv2/tests/test_cached_encoder_inference.py` |
| Computational cost & runtime profiling | `paper/efficiency/`, `paper/figures/make_cost_table.py` |
| Official-protocol CellMap evaluation | `scripts/cellmap_evaluation_pipeline.py`, `nnunetv2/evaluation/cellmap_challenge.py` |

## Subdirectories in `paper/`

- **`reconstruction/`**:
  - `decode_consensus_v2.py`: Implements `Gmix0.5` (Eq. 7), Two-Stage Hierarchical Bayes Decoding (`TS_w0.5`, Eq. 8), and Dual-Weight Consensus-Veto Hysteresis (`TS_w0.5_g0.75_veto`, Eqs. 9–10).
  - `evaluate_cv5_consensus.py`: Evaluates two-stage and consensus-veto decoding across the 5-fold cross-validation dump (`folds 0..4`).
  - `decode_factorial_controls.py` & `decode_lib.py`: Offline decoder for the representation $\times$ group-head factorial and $(w, A)$ hysteresis grid.
- **`analyses/`**:
  - `paper_numbers.py`: Recomputes every number in the manuscript on the 16-crop primary benchmark and 17-crop sensitivity split from per-crop official CellMap evaluation JSONs.
  - `reliability_stats.py` & `robustness_stats.py`: Paired crop bootstrap confidence intervals ($B=10{,}000$), Wilcoxon signed-rank tests, absent-class hallucination audits on 82 object-free validation pairs, and count/volume calibration errors.
  - `confusion_extended.py`: Block-normalized confusion matrices and ribo/cyto confusion attribution.
  - `uniform_filter.py`: Uniform minimum-component-size post-processing sweep across methods.
- **`figures/`**:
  - `make_fig01.py`, `make_fig02_thumbs.py`, `fig02_method.tex`, `make_figures.py`, `make_label_figures.py`, `make_cost_table.py`: Generates Figs. 1–7 and LaTeX tables.
- **`efficiency/`**:
  - `efficiency_metrics.py`, `benchmark_inference.py`, `single_group_timing.py`, `methods.py`: Measures parameter count, FLOPs per $160^3$ tile, peak GPU training memory, and per-volume inference wall-clock times.
