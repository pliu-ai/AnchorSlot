"""Flat nnU-Net (nnUNetTrainerRalph recipe) plus the AnchorSlot coarse group head (reviewer control, 2026-10-27).

The flat network keeps its 32-class softmax output and loss unchanged. A 1x1x1 conv on the last decoder features
predicts the same 18 coarse classes as the AnchorSlot group head (background, the four anchors ecs/pm/cyto/nucpl,
and the 13 dynamic groups), trained on every voxel with the same CE + soft Dice (background excluded) at weight
NNUNET_FLATGH_LAMBDA_GROUP (default 0.5, the AnchorSlot value). The LUT and loss are verbatim copies of
trainer_structured_conditional_er_dynamic_group_head.py so both models see the identical coarse target.

The head wraps decoder.seg_layers[-1], so its parameters take part in every forward pass (DDP-safe) and the
predictor builds the same module tree through build_network_architecture. With NNUNET_FLATGH_APPEND_GROUP_LOGITS=1
the 18 group logits are concatenated after the 32 flat logits (50 channels) for the dump predictor; otherwise the
network output is exactly the flat output, so validation and the built-in final validation are unchanged.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import autocast, nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerRalph import nnUNetTrainerRalph
from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.label_mapping_er_dynamic import (
    DYNAMIC_GROUP_SPECS, FIXED_OUTPUT_TO_ORIGINAL)
from nnunetv2.utilities.helpers import dummy_context

NUM_GROUP_HEAD_CLASSES = 5 + len(DYNAMIC_GROUP_SPECS)
IGNORE_INDEX = -100
_ANCHOR_LABELS = [int(label) for _, label in sorted(FIXED_OUTPUT_TO_ORIGINAL.items())]   # ecs, pm, cyto, nucpl


def build_group_lut() -> torch.Tensor:
    """Original CellMap label id (0..31) -> coarse class: 0 background, 1-4 anchors, 5+g dynamic group g."""
    lut = torch.full((32,), -1, dtype=torch.long)
    lut[0] = 0
    for i, label in enumerate(_ANCHOR_LABELS):
        lut[label] = 1 + i
    for spec in DYNAMIC_GROUP_SPECS:
        for label in spec.original_labels:
            lut[int(label)] = 5 + int(spec.group_id)
    if int((lut < 0).sum()) != 0:
        raise RuntimeError(f"group LUT leaves labels unmapped: {torch.nonzero(lut < 0).flatten().tolist()}")
    return lut


def group_targets(target: torch.Tensor, valid: Optional[torch.Tensor], lut: torch.Tensor) -> torch.Tensor:
    """B x 1 x ... original labels -> B x ... coarse labels; negative labels and invalid voxels -> IGNORE_INDEX."""
    t = target[:, 0]
    invalid = t < 0
    if valid is not None:
        invalid = invalid | ~(valid[:, 0] > 0)
    coarse = lut.to(t.device)[t.clamp(min=0).long()]
    return coarse.masked_fill(invalid, IGNORE_INDEX)


def group_head_loss(logits: torch.Tensor, target: torch.Tensor, valid: Optional[torch.Tensor], lut: torch.Tensor,
                    ce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1e-5) -> torch.Tensor:
    """CE + per-sample soft Dice (background excluded) over the 18 coarse classes, restricted to valid voxels."""
    coarse = group_targets(target, valid, lut)
    mask = coarse != IGNORE_INDEX
    n_valid = mask.sum()
    if int(n_valid) == 0:
        return logits.sum() * 0.0
    ce = F.cross_entropy(logits, coarse, ignore_index=IGNORE_INDEX, reduction="sum") / n_valid
    m = mask.unsqueeze(1).to(logits.dtype)
    probs = torch.softmax(logits, dim=1) * m
    onehot = torch.zeros_like(probs).scatter_(1, coarse.clamp(min=0).unsqueeze(1), 1.0) * m
    axes = tuple(range(2, logits.ndim))
    tp = (probs * onehot).sum(axes)
    fp = probs.sum(axes) - tp
    fn = onehot.sum(axes) - tp
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    dice_loss = 1 - dice[:, 1:].mean()
    return ce_weight * ce + dice_weight * dice_loss


def _flag(name: str) -> bool:
    return str(os.environ.get(name, "0")).lower() in {"1", "true", "yes", "y"}


class SegWithGroupHead(nn.Module):
    """Final segmentation conv plus the coarse group head on the same decoder features."""

    def __init__(self, seg: nn.Module, num_group_classes: int = NUM_GROUP_HEAD_CLASSES):
        super().__init__()
        self.seg = seg
        self.group_head = type(seg)(seg.in_channels, num_group_classes, kernel_size=1, bias=True)
        self.append_group_logits = _flag("NNUNET_FLATGH_APPEND_GROUP_LOGITS")
        self.last_group_logits: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.seg(x)
        group = self.group_head(x)
        if self.append_group_logits:
            return torch.cat((out, group), dim=1)
        self.last_group_logits = group
        return out


class nnUNetTrainerRalphGroupHead(nnUNetTrainerRalph):
    """nnUNetTrainerRalph (flat recipe) + 18-class group head trained on every voxel."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        # explicit signature: nnUNetTrainer records init kwargs by inspecting it
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.lambda_group = float(os.environ.get("NNUNET_FLATGH_LAMBDA_GROUP", "0.5"))
        self.group_lut = build_group_lut()
        self.print_to_log_file(f"[FlatGroupHead] {NUM_GROUP_HEAD_CLASSES}-class group head, "
                               f"lambda_group={self.lambda_group}")

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels, enable_deep_supervision=True):
        net = nnUNetTrainerRalph.build_network_architecture(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import, num_input_channels,
            num_output_channels, enable_deep_supervision=enable_deep_supervision)
        net.decoder.seg_layers[-1] = SegWithGroupHead(net.decoder.seg_layers[-1])
        return net

    def _group_module(self) -> SegWithGroupHead:
        net = self.network.module if self.is_ddp else self.network
        net = getattr(net, "_orig_mod", net)
        return net.decoder.seg_layers[-1]

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = [t.to(self.device, non_blocking=True) for t in target] if isinstance(target, list) \
            else target.to(self.device, non_blocking=True)
        target_high = target[0] if isinstance(target, list) else target

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            flat_loss = self.loss(output, target)
        head = self._group_module()
        group_logits, head.last_group_logits = head.last_group_logits, None
        if group_logits is None:
            raise RuntimeError("group head did not run in the forward pass")
        group_loss = group_head_loss(group_logits.float(), target_high, None, self.group_lut)
        loss = flat_loss + self.lambda_group * group_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": loss.detach().cpu().numpy()}
