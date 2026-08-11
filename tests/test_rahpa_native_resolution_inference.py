from pathlib import Path

from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from scripts.predict_structured_conditional import _build_arg_parser
from scripts.run_rahpa_32nm_gb001 import _prediction_command


def test_native_32nm_spacing_does_not_expand_the_volume() -> None:
    native_shape = compute_new_shape((400, 400, 400), (32.0, 32.0, 32.0), (32.0, 32.0, 32.0))
    wrong_high_resolution_shape = compute_new_shape(
        (400, 400, 400),
        (32.0, 32.0, 32.0),
        (4.0, 4.0, 4.0),
    )

    assert tuple(native_shape) == (400, 400, 400)
    assert tuple(wrong_high_resolution_shape) == (3200, 3200, 3200)


def test_predictor_cli_accepts_a_preprocessing_override() -> None:
    args = _build_arg_parser().parse_args(
        [
            "-i",
            "/input",
            "-o",
            "/output",
            "--preprocessing_dataset_dir",
            "/preprocessed/Dataset201_low_res",
            "--preprocessing_configuration",
            "3d_fullres",
        ]
    )

    assert args.preprocessing_dataset_dir == "/preprocessed/Dataset201_low_res"
    assert args.preprocessing_configuration == "3d_fullres"
    assert args.preprocessing_plans_name == "nnUNetPlans"


def test_gb001_command_uses_native_dataset201_preprocessing() -> None:
    command = _prediction_command(
        python=Path("/env/bin/python"),
        repo_root=Path("/repo"),
        input_dir=Path("/input/case"),
        output_dir=Path("/output/case"),
        model_folder=Path("/model"),
        checkpoint="checkpoint_best.pth",
        preprocessing_dataset_dir=Path("/preprocessed/Dataset201_low_res"),
        preprocessing_configuration="3d_fullres",
    )

    dataset_flag = command.index("--preprocessing_dataset_dir")
    configuration_flag = command.index("--preprocessing_configuration")
    voxel_size_flag = command.index("--voxel_size_nm")
    assert command[dataset_flag + 1] == "/preprocessed/Dataset201_low_res"
    assert command[configuration_flag + 1] == "3d_fullres"
    assert command[voxel_size_flag + 1] == "32"
