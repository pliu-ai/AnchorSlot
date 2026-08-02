from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch
from torch import nn

from anchor_slot.grouping.validate_grouping import (
    DEFAULT_DYNAMIC_LABELS,
    validate_grouping_config,
)
from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.label_mapping_er_dynamic import (
    DYNAMIC_GROUP_SPECS,
)
from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.condition_encoding import (
    TextConditionedGroupEmbedding,
)
from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.network_structured_conditional import (
    StructuredConditionalUNet,
)
from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.label_mapping_er_dynamic import (
    structured_prediction_to_original_labels,
)
from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.text_contrastive import (
    VisualTextContrastiveLoss,
)
from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.text_generated_slot_head import (
    TextGeneratedSlotHead,
)


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Conv3d(1, 2, kernel_size=1),
                nn.Conv3d(2, 4, kernel_size=1),
            ]
        )


class _Decoder(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.deep_supervision = False
        self.transpconvs = nn.ModuleList([nn.ConvTranspose3d(4, 2, kernel_size=1)])
        self.stages = nn.ModuleList([nn.Conv3d(4, 2, kernel_size=1)])
        self.seg_layers = nn.ModuleList([nn.Conv3d(2, out_channels, kernel_size=1)])


class _Backbone(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.decoder = _Decoder(out_channels)


def test_text_conditioned_embedding_shape() -> None:
    encoder = TextConditionedGroupEmbedding(
        num_groups=13,
        cond_dim=64,
        group_text_matrix=torch.randn(13, 32),
        use_text_conditioning=True,
        text_fusion="concat_mlp",
    )
    cond_e = encoder(torch.tensor([0, 12]), batch_size=2, device=torch.device("cpu"))
    assert cond_e.shape == (2, 64)


def test_text_conditioned_embedding_requires_text_matrix() -> None:
    with pytest.raises(ValueError, match="requires group_text_matrix"):
        TextConditionedGroupEmbedding(num_groups=13, cond_dim=64, use_text_conditioning=True)


def test_structured_network_output_shape_learned_and_text_conditioned() -> None:
    x = torch.randn(2, 1, 8, 8, 8)
    group_ids = torch.tensor([0, 12])
    learned = StructuredConditionalUNet(
        backbone=_Backbone(out_channels=9),
        num_groups=13,
        num_output_channels=9,
        cond_dim=64,
        condition_mode="learned",
    )
    text_conditioned = StructuredConditionalUNet(
        backbone=_Backbone(out_channels=9),
        num_groups=13,
        num_output_channels=9,
        cond_dim=64,
        condition_mode="learned_text",
        group_text_matrix=torch.randn(13, 32),
        text_fusion="concat_mlp",
    )

    assert learned(x, group_ids).shape == (2, 9, 8, 8, 8)
    assert text_conditioned(x, group_ids).shape == (2, 9, 8, 8, 8)


def test_visual_text_contrastive_loss_finite_scalar() -> None:
    loss_fn = VisualTextContrastiveLoss(text_contrast_num_samples=8)
    features = torch.randn(1, 16, 4, 4, 4)
    labels = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
    labels[:, :, :4] = 1
    labels[:, :, 4:] = 4
    text = torch.randn(32, 16)
    loss = loss_fn(
        feature_map=features,
        original_label_map=labels,
        group_ids=torch.tensor([0]),
        text_embeddings_by_label=text,
        valid_original_label_ids=torch.tensor([1, 2, 3, 4, 5, 6, 27]),
        active_group_label_mapping={0: (4, 5, 6)},
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss_fn.last_stats["num_samples"] > 0


def test_visual_text_contrastive_loss_no_valid_samples_returns_zero() -> None:
    loss_fn = VisualTextContrastiveLoss()
    features = torch.randn(1, 16, 4, 4, 4)
    labels = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
    text = torch.randn(32, 16)
    loss = loss_fn(
        feature_map=features,
        original_label_map=labels,
        group_ids=torch.tensor([0]),
        text_embeddings_by_label=text,
        valid_original_label_ids=torch.tensor([1, 4]),
        active_group_label_mapping={0: (4, 5, 6)},
    )
    assert torch.isfinite(loss)
    assert float(loss.item()) == 0.0
    assert loss_fn.last_stats["num_samples"] == 0


def test_visual_text_contrastive_default_selection_excludes_background_and_nonquery_present_labels() -> None:
    loss_fn = VisualTextContrastiveLoss(text_contrast_num_samples=8)
    features = torch.randn(1, 16, 4, 4, 4)
    labels = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
    labels[:, :, :2] = 0
    labels[:, :, 2:5] = 4
    labels[:, :, 5:] = 7
    text = torch.randn(32, 16)
    loss = loss_fn(
        feature_map=features,
        original_label_map=labels,
        group_ids=torch.tensor([0]),
        text_embeddings_by_label=text,
        valid_original_label_ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 27]),
        active_group_label_mapping={0: (4, 5, 6), 1: (7, 8)},
    )
    assert torch.isfinite(loss)
    sampled = set(loss_fn.last_stats["sampled_label_ids"])
    assert 0 not in sampled
    assert 7 not in sampled
    assert 4 in sampled


def test_text_contrastive_one_step_backward_and_logits_shape_unchanged() -> None:
    model = StructuredConditionalUNet(
        backbone=_Backbone(out_channels=9),
        num_groups=13,
        num_output_channels=9,
        cond_dim=64,
        condition_mode="learned",
        use_text_contrastive=True,
        text_contrast_text_dim=8,
        text_contrast_dim=4,
    )
    loss_fn = VisualTextContrastiveLoss(text_contrast_num_samples=4)
    x = torch.randn(2, 1, 8, 8, 8)
    group_ids = torch.tensor([0, 12])
    labels = torch.zeros(2, 1, 8, 8, 8, dtype=torch.long)
    labels[0, :, :4] = 4
    labels[1, :, 4:] = 17
    text = torch.randn(32, 8)

    output_pack = model(x, group_ids, return_features=True)
    logits = output_pack["logits"]
    projected_features = model.visual_text_proj(output_pack["decoder_last"])
    projected_text = model.text_contrast_proj(text)
    assert logits.shape == (2, 9, 8, 8, 8)

    loss = logits.mean() + loss_fn(
        feature_map=projected_features,
        original_label_map=labels,
        group_ids=group_ids,
        text_embeddings_by_label=projected_text,
        valid_original_label_ids=torch.tensor([1, 2, 3, 4, 5, 6, 17, 18, 27]),
        active_group_label_mapping={0: (4, 5, 6), 12: (17, 18)},
    )
    loss.backward()
    assert any(p.grad is not None for p in model.visual_text_proj.parameters())
    assert any(p.grad is not None for p in model.text_contrast_proj.parameters())


def test_text_generated_slot_head_group0_uses_mito_slot_order() -> None:
    head = TextGeneratedSlotHead(
        in_channels=3,
        text_embedding_dim=3,
        num_slots=3,
        hidden_dim=4,
        use_bias=False,
        normalize_weight=False,
    )
    head.weight_mlp = nn.Identity()
    features = torch.zeros(1, 3, 2, 2, 2)
    features[:, 0] = 1.0
    features[:, 1] = 2.0
    features[:, 2] = 3.0
    embeddings = {
        "mito_mem": torch.tensor([1.0, 0.0, 0.0]),
        "mito_lum": torch.tensor([0.0, 1.0, 0.0]),
        "mito_ribo": torch.tensor([0.0, 0.0, 1.0]),
    }
    group_to_labels = {0: ("mito_mem", "mito_lum", "mito_ribo")}

    logits = head(features, torch.tensor([0]), embeddings, group_to_labels)

    assert logits.shape == (1, 3, 2, 2, 2)
    assert torch.allclose(logits[:, 0], torch.ones_like(logits[:, 0]) * 1.0)
    assert torch.allclose(logits[:, 1], torch.ones_like(logits[:, 1]) * 2.0)
    assert torch.allclose(logits[:, 2], torch.ones_like(logits[:, 2]) * 3.0)


def test_text_generated_slot_head_two_label_group_masks_third_slot() -> None:
    head = TextGeneratedSlotHead(
        in_channels=2,
        text_embedding_dim=2,
        num_slots=3,
        hidden_dim=4,
        use_bias=False,
        normalize_weight=False,
    )
    head.weight_mlp = nn.Identity()
    features = torch.randn(1, 2, 2, 2, 2)
    embeddings = {
        "golgi_mem": torch.tensor([1.0, 0.0]),
        "golgi_lum": torch.tensor([0.0, 1.0]),
    }
    group_to_labels = {1: ("golgi_mem", "golgi_lum")}

    logits = head(features, torch.tensor([1]), embeddings, group_to_labels)

    assert logits.shape == (1, 3, 2, 2, 2)
    assert torch.all(logits[:, 2] <= -9999.0)
    assert head.last_stats["active_slots"] == 2
    assert head.last_stats["inactive_slots"] == 1


def test_structured_network_text_generated_slot_head_shape_and_inactive_slot() -> None:
    group_slot_text = torch.randn(13, 3, 8)
    group_to_labels = {0: ("mito_mem", "mito_lum", "mito_ribo"), 1: ("golgi_mem", "golgi_lum")}
    model = StructuredConditionalUNet(
        backbone=_Backbone(out_channels=9),
        num_groups=13,
        num_output_channels=9,
        cond_dim=64,
        condition_mode="learned",
        use_text_generated_slot_head=True,
        text_generated_slot_embeddings=group_slot_text,
        text_generated_slot_group_to_labels=group_to_labels,
        text_generated_slot_hidden_dim=16,
        text_generated_slot_mode="replace",
    )
    x = torch.randn(2, 1, 8, 8, 8)
    logits = model(x, torch.tensor([0, 1]))

    assert logits.shape == (2, 9, 8, 8, 8)
    assert torch.all(logits[1, 7] <= -9999.0)


def test_text_generated_slot_head_one_step_backward() -> None:
    group_slot_text = torch.randn(13, 3, 8)
    group_to_labels = {0: ("mito_mem", "mito_lum", "mito_ribo"), 12: ("er_mem", "er_lum")}
    model = StructuredConditionalUNet(
        backbone=_Backbone(out_channels=9),
        num_groups=13,
        num_output_channels=9,
        cond_dim=64,
        condition_mode="learned",
        use_text_generated_slot_head=True,
        text_generated_slot_embeddings=group_slot_text,
        text_generated_slot_group_to_labels=group_to_labels,
        text_generated_slot_hidden_dim=16,
        text_generated_slot_mode="residual",
        text_generated_slot_alpha=1.0,
    )
    x = torch.randn(2, 1, 8, 8, 8)
    logits = model(x, torch.tensor([0, 12]))
    loss = logits[:, :8].mean()
    loss.backward()

    assert logits.shape == (2, 9, 8, 8, 8)
    assert any(p.grad is not None for p in model.text_generated_slot_heads[-1].parameters())


def test_reconstruction_still_maps_dynamic_slots_by_group() -> None:
    pred = torch.tensor([[[[5, 6, 7]]]], dtype=torch.long)
    reconstructed = structured_prediction_to_original_labels(pred, group_id=0)
    assert reconstructed.tolist() == [[[[4, 5, 6]]]]


def _manual_grouping_config() -> dict:
    return {
        "anchors": ["ecs", "pm", "cyto", "nucpl"],
        "dynamic_groups": {
            "G1": ["mito_mem", "mito_lum", "mito_ribo"],
            "G2": ["golgi_mem", "golgi_lum"],
            "G3": ["ves_mem", "ves_lum"],
            "G4": ["endo_mem", "endo_lum"],
            "G5": ["lyso_mem", "lyso_lum"],
            "G6": ["ld_mem", "ld_lum"],
            "G7": ["eres_mem", "eres_lum"],
            "G8": ["hchrom", "echrom"],
            "G9": ["ne_mem", "ne_lum"],
            "G10": ["np_out", "np_in"],
            "G11": ["mt_out", "mt_in"],
            "G12": ["perox_mem", "perox_lum"],
            "G13": ["er_mem", "er_lum"],
        },
        "K": 3,
    }


def test_default_grouping_unchanged() -> None:
    expected = tuple(tuple(group) for group in _manual_grouping_config()["dynamic_groups"].values())
    actual = tuple(tuple(spec.subclass_names) for spec in DYNAMIC_GROUP_SPECS)
    assert actual == expected


def test_generated_grouping_passes_validation() -> None:
    config = {
        "anchors": ["ecs", "pm", "cyto", "nucpl"],
        "dynamic_groups": {
            "G1": ["mito_mem", "mito_lum", "mito_ribo"],
            "G2": ["golgi_mem", "golgi_lum", "ves_mem"],
            "G3": ["ves_lum", "endo_mem", "endo_lum"],
            "G4": ["lyso_mem", "lyso_lum", "ld_mem"],
            "G5": ["ld_lum", "er_mem", "er_lum"],
            "G6": ["eres_mem", "eres_lum", "ne_mem"],
            "G7": ["ne_lum", "np_out", "np_in"],
            "G8": ["hchrom", "echrom", "mt_out"],
            "G9": ["mt_in", "perox_mem", "perox_lum"],
        },
        "K": 3,
    }
    result = validate_grouping_config(config)
    assert result["num_dynamic_labels"] == len(DEFAULT_DYNAMIC_LABELS)
    assert result["max_group_size"] == 3


def test_grouping_validation_missing_label_triggers_clear_error() -> None:
    config = _manual_grouping_config()
    config["dynamic_groups"]["G1"] = ["mito_mem", "mito_lum"]
    with pytest.raises(ValueError, match="missing required dynamic labels"):
        validate_grouping_config(config)


def test_grouping_validation_duplicate_label_triggers_clear_error() -> None:
    config = _manual_grouping_config()
    config["dynamic_groups"]["G2"] = ["golgi_mem", "golgi_lum", "mito_ribo"]
    with pytest.raises(ValueError, match="duplicate dynamic labels"):
        validate_grouping_config(config)


def test_grouping_validation_group_size_triggers_clear_error() -> None:
    config = _manual_grouping_config()
    config["dynamic_groups"]["G2"] = ["golgi_mem", "golgi_lum", "ves_mem", "ves_lum"]
    with pytest.raises(ValueError, match="max_group_size"):
        validate_grouping_config(config)


def test_training_import_can_initialize_generated_grouping_config(tmp_path) -> None:
    config = _manual_grouping_config()
    config["dynamic_groups"] = {
        "G1": ["mito_mem", "mito_lum", "mito_ribo"],
        "G2": ["golgi_mem", "golgi_lum", "ves_mem"],
        "G3": ["ves_lum", "endo_mem", "endo_lum"],
        "G4": ["lyso_mem", "lyso_lum", "ld_mem"],
        "G5": ["ld_lum", "er_mem", "er_lum"],
        "G6": ["eres_mem", "eres_lum", "ne_mem"],
        "G7": ["ne_lum", "np_out", "np_in"],
        "G8": ["hchrom", "echrom", "mt_out"],
        "G9": ["mt_in", "perox_mem", "perox_lum"],
    }
    path = tmp_path / "grouping.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = "/projects/weilab/liupeng/code/vendors/nnUNet:/projects/weilab/liupeng/code/projects/RAHSeg"
    env["NNUNET_STRUCTCOND_GROUPING_CONFIG_PATH"] = str(path)
    code = (
        "from nnunetv2.training.nnUNetTrainer.variants.structured_conditional import label_mapping_er_dynamic as lm; "
        "assert lm.NUM_DYNAMIC_GROUPS == 9; "
        "assert lm.get_group_spec(1).subclass_names == ('golgi_mem', 'golgi_lum', 'ves_mem')"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
