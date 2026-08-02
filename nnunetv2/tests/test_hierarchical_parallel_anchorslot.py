from __future__ import annotations

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.hierarchical_parallel_mapping import (
    GROUP_COARSE_START,
    NUM_COARSE_CHANNELS,
    build_hierarchical_targets,
)
from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.network_hierarchical_parallel_anchorslot import (
    HierarchicalParallelAnchorSlotUNet,
)
from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.structured_loss_hierarchical_parallel_anchorslot import (
    HierarchicalParallelAnchorSlotLoss,
)


class _TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU()),
                nn.Sequential(nn.Conv2d(4, 8, 3, stride=2, padding=1), nn.ReLU()),
                nn.Sequential(nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU()),
            ]
        )


class _TinyDecoder(nn.Module):
    def __init__(self, deep_supervision: bool) -> None:
        super().__init__()
        self.transpconvs = nn.ModuleList(
            [nn.ConvTranspose2d(16, 8, 2, stride=2), nn.ConvTranspose2d(8, 4, 2, stride=2)]
        )
        self.stages = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(16, 8, 3, padding=1), nn.ReLU()),
                nn.Sequential(nn.Conv2d(8, 4, 3, padding=1), nn.ReLU()),
            ]
        )
        self.seg_layers = nn.ModuleList(
            [nn.Conv2d(8, NUM_COARSE_CHANNELS, 1), nn.Conv2d(4, NUM_COARSE_CHANNELS, 1)]
        )
        self.deep_supervision = deep_supervision


class _TinyBackbone(nn.Module):
    def __init__(self, deep_supervision: bool = False) -> None:
        super().__init__()
        self.encoder = _TinyEncoder()
        self.decoder = _TinyDecoder(deep_supervision)


def test_hierarchy_target_mapping_for_fixed_and_dynamic_labels() -> None:
    target = torch.tensor([[[[0, 1, 4, 5, 30, 31, 255]]]])
    semantic, coarse, group, slot, valid = build_hierarchical_targets(target, ignore_label=255)

    assert semantic[0, 0, 0, :6].tolist() == [0, 1, 4, 5, 30, 31]
    assert coarse[0, 0, 0, 2:6].tolist() == [GROUP_COARSE_START, GROUP_COARSE_START, 19, 19]
    assert group[0, 0, 0, 2:6].tolist() == [0, 0, 11, 11]
    assert slot[0, 0, 0, 2:6].tolist() == [0, 1, 0, 1]
    assert valid[0, 0, 0].tolist() == [True, True, True, True, True, True, False]


def test_parallel_network_produces_normalized_32_class_distribution() -> None:
    network = HierarchicalParallelAnchorSlotUNet(_TinyBackbone(), code_dim=16)
    image = torch.randn(2, 1, 16, 16)
    output = network(image, return_hierarchy=True)

    assert output["semantic_logits"].shape == (2, 32, 16, 16)
    assert output["coarse_logits"].shape == (2, NUM_COARSE_CHANNELS, 16, 16)
    assert output["slot_logits"].shape == (2, 12, 2, 16, 16)
    log_mass = torch.logsumexp(output["semantic_logits"], dim=1)
    torch.testing.assert_close(log_mass, torch.zeros_like(log_mass), atol=1e-5, rtol=1e-5)


def test_hierarchy_composition_promotes_mixed_precision_inputs() -> None:
    network = HierarchicalParallelAnchorSlotUNet(_TinyBackbone(), code_dim=8)
    coarse_logits = torch.randn(1, NUM_COARSE_CHANNELS, 4, 4, dtype=torch.float16)
    slot_logits = torch.randn(1, 12, 2, 4, 4, dtype=torch.float32)

    semantic = network.compose_semantic_log_probs(coarse_logits, slot_logits)

    assert semantic.dtype == torch.float32
    assert torch.isfinite(semantic).all()
    torch.testing.assert_close(
        torch.logsumexp(semantic, dim=1),
        torch.zeros_like(semantic[:, 0]),
        atol=1e-4,
        rtol=1e-4,
    )


def test_deep_supervision_keeps_hierarchy_outputs_aligned() -> None:
    network = HierarchicalParallelAnchorSlotUNet(_TinyBackbone(deep_supervision=True), code_dim=8)
    output = network(torch.randn(1, 1, 8, 8), return_hierarchy=True)
    assert all(isinstance(output[key], list) for key in output)
    assert len(output["semantic_logits"]) == len(output["coarse_logits"]) == len(output["slot_logits"]) == 2
    assert output["semantic_logits"][0].shape[-2:] == (8, 8)
    assert output["semantic_logits"][1].shape[-2:] == (4, 4)


def test_joint_hierarchical_loss_backpropagates_to_group_and_anchor_codes() -> None:
    network = HierarchicalParallelAnchorSlotUNet(_TinyBackbone(), code_dim=16)
    image = torch.randn(1, 1, 8, 8)
    target = torch.randint(0, 32, (1, 1, 8, 8))
    output = network(image, return_hierarchy=True)
    loss = HierarchicalParallelAnchorSlotLoss()(
        output["semantic_logits"],
        output["coarse_logits"],
        output["slot_logits"],
        target,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert network.group_embeddings.grad is not None
    assert network.anchor_slot_embeddings.grad is not None
    assert torch.isfinite(network.group_embeddings.grad).all()
    assert torch.isfinite(network.anchor_slot_embeddings.grad).all()
