# AnchorSlot: Role-Structured Supervision for Pan-Organelle Segmentation in Volume EM

Official implementation for **"AnchorSlot: Role-Structured Supervision for Pan-Organelle Segmentation in Volume EM"** (Peng Liu, Jiancheng Yang, and Donglai Wei), built on nnU-Net v2.

AnchorSlot factorizes 31-class partially labeled connectomics segmentation into fixed context channels and query-specific organelle slots conditioned by Feature-wise Linear Modulation (FiLM):
- **Background + 4 Fixed Context Anchors**: Always-supervised spatial landmarks (`ecs`, `pm`, `mito_mem`, `mito_lum`).
- **$K=3$ Reusable Dynamic Organelle Slots**: Query-conditioned slots resolving membrane, lumen, and sub-compartment roles for whichever of the 13 organelle groups $g \in \{0, \dots, 12\}$ is queried.
- **Explicit Non-Query Foreground (`other`)**: Absorbs annotated foreground belonging to non-queried organelle groups so absent-group annotations provide valid negative evidence without teaching background on other organelles.
- **18-Class Coarse Group Head**: Shared unconditioned decoder head predicting background, the 4 anchors, and the 13 dynamic organelle groups in one pass for coarse-to-fine geometric fusion and two-stage consensus-veto reconstruction.

---

## Repository Structure

### 1. Core AnchorSlot Architecture & Trainers (`nnunetv2/training/nnUNetTrainer/variants/structured_conditional/`)
- [`network_structured_conditional.py`](nnunetv2/training/nnUNetTrainer/variants/structured_conditional/network_structured_conditional.py): `StructuredConditionalUNet` with FiLM query conditioning at every decoder stage.
- [`label_mapping_er_dynamic.py`](nnunetv2/training/nnUNetTrainer/variants/structured_conditional/label_mapping_er_dynamic.py): Role-structured label partition (4 fixed anchors, 13 dynamic query groups, $K=3$ slots, non-query foreground channel).
- [`dataloader_structured_conditional_er_dynamic.py`](nnunetv2/training/nnUNetTrainer/variants/structured_conditional/dataloader_structured_conditional_er_dynamic.py): Partial-label aware 3D patch sampler with 80/20 positive/annotated-negative query sampling.
- [`structured_loss_er_dynamic.py`](nnunetv2/training/nnUNetTrainer/variants/structured_conditional/structured_loss_er_dynamic.py): Masked deep-supervision Dice + cross-entropy + anchor consistency loss.
- [`trainer_structured_conditional_er_dynamic.py`](nnunetv2/training/nnUNetTrainer/variants/structured_conditional/trainer_structured_conditional_er_dynamic.py): Base AnchorSlot trainer (`nnUNetTrainerStructuredConditionalERDynamic`).
- [`trainer_structured_conditional_er_dynamic_group_head.py`](nnunetv2/training/nnUNetTrainer/variants/structured_conditional/trainer_structured_conditional_er_dynamic_group_head.py): **AnchorSlot + Group Head** trainer (`nnUNetTrainerStructuredConditionalERDynamicGroupHead`) attaching the 18-class coarse group head ($\lambda_{\text{gh}} = 0.3$).
- [`trainer_structured_conditional_er_dynamic_group_head_nqbg.py`](nnunetv2/training/nnUNetTrainer/variants/structured_conditional/trainer_structured_conditional_er_dynamic_group_head_nqbg.py): Non-query-as-background factorial ablation trainer (`nnUNetTrainerStructuredConditionalERDynamicGroupHeadNQBG`).
- [`trainer_structured_conditional_er_dynamic_ablations.py`](nnunetv2/training/nnUNetTrainer/variants/structured_conditional/trainer_structured_conditional_er_dynamic_ablations.py): Controlled training ablations (100%-positive query sampling `Pos100`, unmasked loss `Unmasked`, and unweighted CE `Unweighted`).
- [`trainer_structured_conditional_er_dynamic_ddw.py`](nnunetv2/training/nnUNetTrainer/variants/structured_conditional/trainer_structured_conditional_er_dynamic_ddw.py): Dynamic distribution weighting fine-tuning trainer for the official CellMap test-server submission.
- [`trainer_structured_conditional_er_dynamic_mednext.py`](nnunetv2/training/nnUNetTrainer/variants/structured_conditional/trainer_structured_conditional_er_dynamic_mednext.py) & [`trainer_mednext_32out_no_cond.py`](nnunetv2/training/nnUNetTrainer/variants/structured_conditional/trainer_mednext_32out_no_cond.py): Matched MedNeXt-B backbone trainers (AnchorSlot vs. Flat 32-class).
- [`trainer_synapse_reconstruction.py`](nnunetv2/training/nnUNetTrainer/variants/structured_conditional/trainer_synapse_reconstruction.py): External transfer trainer on multi-organ abdominal CT (Synapse CT).

### 2. Inference, Geometric Fusion & Two-Stage Consensus-Veto Reconstruction
- [`scripts/predict_structured_conditional.py`](scripts/predict_structured_conditional.py): Sliding-window predictor supporting single-pass encoder caching (`--cache_encoder`, $1.47\times$ speedup with identical outputs).
- [`scripts/predict_dump_group_head.py`](scripts/predict_dump_group_head.py) & [`scripts/merge_group_head.py`](scripts/merge_group_head.py): Compact 63-channel probability dump and geometric group-head fusion (`Gmix0.5`, Eq. 7) + 6-connected hysteresis (`HGmix0.5a3`).
- [`paper/reconstruction/decode_consensus_v2.py`](paper/reconstruction/decode_consensus_v2.py): **Two-Stage Hierarchical Bayes Decoding** (`TS_w0.5`) and **Dual-Weight Consensus-Veto Hysteresis** (`TS_w0.5_g0.75_veto`, Eqs. 8–10) eliminating multi-slot vote splitting and vetoing high-conflict false-positive components.
- [`paper/reconstruction/evaluate_cv5_consensus.py`](paper/reconstruction/evaluate_cv5_consensus.py): 5-fold cross-validation evaluator for two-stage consensus-veto decoding.
- [`paper/reconstruction/decode_factorial_controls.py`](paper/reconstruction/decode_factorial_controls.py): Offline decoder for the $2\times 2$ representation $\times$ group-head factorial and $(w, A)$ sensitivity grid.

### 3. Adapted Multi-Organelle & Multi-Dataset Baselines
All adapted baselines share the exact same PlainConvUNet backbone, 202-crop training split, $160^3$ patch size, 4 nm spacing, and 750,000-iteration optimization budget:
- **Flat nnU-Net / Class-Balanced Flat / Flat + Group Head**: [`nnUNetTrainerRalph.py`](nnunetv2/training/nnUNetTrainer/nnUNetTrainerRalph.py), [`nnUNetTrainerRalphGroupHead.py`](nnunetv2/training/nnUNetTrainer/nnUNetTrainerRalphGroupHead.py).
- **Binary / Ternary FiLM**: [`nnUNetTrainerConditionalFiLM.py`](nnunetv2/training/nnUNetTrainer/variants/partial_label/nnUNetTrainerConditionalFiLM.py), [`nnUNetTrainerConditionalFiLMTernary.py`](nnunetv2/training/nnUNetTrainer/variants/partial_label/nnUNetTrainerConditionalFiLMTernary.py).
- **CoNeMOS (adapted), UniSeg (adapted), and MISO-style sampling**: [`nnUNetTrainerLiterature.py`](nnunetv2/training/nnUNetTrainer/nnUNetTrainerLiterature.py), [`literature_baselines/`](literature_baselines/), [`scripts/predict_literature_baselines.py`](scripts/predict_literature_baselines.py).
- **DoDNet (adapted), Blob Loss, HSSN, and SALT**: [`nnunetv2/training/nnUNetTrainer/variants/comparison_baselines/`](nnunetv2/training/nnUNetTrainer/variants/comparison_baselines/), [`scripts/run_comparison_baseline.py`](scripts/run_comparison_baseline.py).

### 4. Data Splits, Official CellMap Evaluation & Paper Reproducibility
- [`configs/cellmap/source_roi_audit_16crop.json`](configs/cellmap/source_roi_audit_16crop.json): Source-volume and 3D bounding-box audit of the 219 CellMap crops and the clean 16-crop primary validation benchmark (excluding the zero-pad-overlapping `crop126`).
- [`scripts/cellmap_evaluation_pipeline.py`](scripts/cellmap_evaluation_pipeline.py) & [`nnunetv2/evaluation/cellmap_challenge.py`](nnunetv2/evaluation/cellmap_challenge.py): Official CellMap Zarr-2 submission exporter, validator, and pinned `0300239` evaluator wrapper.
- [`paper/`](paper/README.md): Scripts reproducing every table, statistical test, bootstrap confidence interval, confusion matrix, and figure in the TMI manuscript.

---

## Installation

```bash
conda env create -f environment.yml
conda activate anchorslot
pip install -e '.[cellmap]'
```

Configure standard nnU-Net environment variables:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

---

## Training & Inference

### Train AnchorSlot + Group Head
```bash
nnUNetv2_train 205 3d_lowres_large_patch 0 \
  -tr nnUNetTrainerStructuredConditionalERDynamicGroupHead \
  -p nnUNetPlans
```

### Run Encoder-Cached 13-Query Inference & Group-Head Dump
```bash
python scripts/predict_structured_conditional.py \
  -i "$nnUNet_raw/Dataset205_CellMap_fill/imagesTs" \
  -o /path/to/predictions \
  -m /path/to/model_folder \
  -f 0 -c 3d_lowres_large_patch \
  --group_id all --cache_encoder
```

### Run Two-Stage & Consensus-Veto Reconstruction
```bash
python paper/reconstruction/decode_consensus_v2.py \
  --dump /path/to/group_head_dump \
  --model-folder /path/to/model_folder \
  --out-root /path/to/decoded_variants \
  --variants Gmix0.5 TS_w0.5 TS_w0.5_g0.75_veto
```

### Official CellMap Zarr-2 Export & Validation
```bash
anchorslot_export_cellmap_submission \
  --predictions /path/to/official_test_predictions \
  --manifest /path/to/test_crop_manifest.csv \
  --output /path/to/submission.zarr

anchorslot_validate_cellmap_submission \
  --submission /path/to/submission.zarr \
  --manifest /path/to/test_crop_manifest.csv
```

---

## Running Unit Tests

All unit tests run on CPU without requiring CellMap image volumes:

```bash
pytest -q nnunetv2/tests/test_anchorslot_group_head.py \
          nnunetv2/tests/test_anchorslot_training_ablations.py \
          nnunetv2/tests/test_cached_encoder_inference.py \
          nnunetv2/tests/test_literature_baselines.py \
          nnunetv2/tests/test_misostyle.py \
          nnunetv2/tests/test_comparison_blob_loss.py \
          nnunetv2/tests/test_comparison_dodnet.py \
          nnunetv2/tests/test_comparison_trainers.py \
          paper/reconstruction/test_decode_consensus_v2.py \
          paper/reconstruction/test_factorial_controls.py
```

---

## Citation & License

This repository is distributed under the Apache-2.0 License (see [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)). If you use AnchorSlot in your research, please cite our paper and nnU-Net (see [`CITATION.cff`](CITATION.cff)).
