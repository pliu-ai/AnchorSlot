"""The three CellMap models compared in the efficiency table (paper checkpoints; weights do not affect cost)."""

from __future__ import annotations

from pathlib import Path

RUN = Path(__file__).resolve().parent
RESULTS = Path("/projects/weilab/liupeng/data/nnUNet/nnUNet_results")
CONFIGURATION = "3d_lowres_large_patch"
PATCH = (160, 160, 160)

# Training environment of the primary AnchorSlot recipe (4-GPU run, global batch 4).
ANCHORSLOT_TRAIN_ENV = {
    "NNUNET_STRUCTCOND_COMPILE": "0", "NNUNET_USE_WANDB": "0",
    "NNUNET_STRUCTCOND_INITIAL_LR": "0.001", "NNUNET_STRUCTCOND_WEIGHT_DECAY": "0.00003",
    "NNUNET_STRUCTCOND_P_PRESENT_GROUP": "0.8", "NNUNET_STRUCTCOND_GROUP_SEED": "1234",
    "NNUNET_STRUCTCOND_GROUP_SAMPLING_MODE": "random",
    "NNUNET_STRUCTCOND_LAMBDA_CE": "1", "NNUNET_STRUCTCOND_LAMBDA_DICE": "1",
    "NNUNET_STRUCTCOND_LAMBDA_COND": "0.25", "NNUNET_STRUCTCOND_LAMBDA_SUPPRESS": "0.1",
    "NNUNET_STRUCTCOND_ENABLE_COND": "1", "NNUNET_STRUCTCOND_ENABLE_SUPPRESS": "1",
    "NNUNET_STRUCTCOND_SKIP_NONFINITE_GRAD": "1",
}

METHODS = {
    "flat": {
        "model_dir": str(RESULTS / "Dataset200_Dataset101_CellMap_model_high_res"
                         / "nnUNetTrainer__nnUNetPlans__3d_lowres_large_patch"),
        "fold": "all", "checkpoint": "checkpoint_final.pth", "trainer": "nnUNetTrainer", "structured": False,
    },
    "cb_flat": {
        "model_dir": str(RESULTS / "rahseg/flat_baselines/flat31_class_balanced_sqrt_ce_init_lr0.001_stable"
                         / "Dataset202_Dataset101_CellMap_model_high_res"
                         / "nnUNetTrainerRalph__nnUNetPlans__3d_lowres_large_patch"),
        "fold": 0, "checkpoint": "checkpoint_best.pth", "trainer": "nnUNetTrainerRalph", "structured": False,
        "train_env": {"RALPH_CONFIG_JSON": str(RUN / "ralph_cb_config.json")},
    },
    "anchorslot": {
        "model_dir": str(RESULTS / "rahseg/structured_conditional/structured_conditional_er_dynamic_from_scratch_init_lr0.001"
                         / "Dataset202_Dataset101_CellMap_model_high_res"
                         / "nnUNetTrainerStructuredConditionalERDynamic__nnUNetPlans__3d_lowres_large_patch"),
        "fold": 0, "checkpoint": "checkpoint_best.pth", "trainer": "nnUNetTrainerStructuredConditionalERDynamic",
        "structured": True, "train_env": ANCHORSLOT_TRAIN_ENV,
    },
}
