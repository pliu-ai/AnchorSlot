"""AnchorSlot with a group-classification head (direction 2b).

The structured head decodes one queried group per forward pass, so voxels of two different groups (e.g. lysosome vs
endosome) never compete in one softmax; at inference the per-group slot probabilities of independent query runs are
merged by an uncalibrated max. This variant adds a 1x1x1 head on the last decoder features that predicts a coarse,
query-independent label: background, the four anchors, or one of the 13 dynamic groups (18 classes). It is trained on
every voxel at every step (CE + soft Dice, weight NNUNET_STRUCTCOND_LAMBDA_GROUP, default 0.5) and, at inference,
averaged over the 13 query runs like the anchors, giving a calibrated P(group | voxel) to arbitrate between groups.

Network modes:
  forward(x, ids)                          -> unchanged structured output (validation and legacy paths untouched)
  forward(x, ids, return_group_logits=True) -> {"logits": structured output, "group_logits": B x 18 x ...}
  NNUNET_STRUCTCOND_APPEND_GROUP_LOGITS=1   -> the 18 group logits are appended to the highest-resolution output
                                              (9 + 18 channels) so the unchanged sliding-window predictor carries them.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .label_mapping_er_dynamic import DYNAMIC_GROUP_SPECS, FIXED_OUTPUT_TO_ORIGINAL
from .network_structured_conditional import StructuredConditionalUNet
from .trainer_structured_conditional_er_dynamic import nnUNetTrainerStructuredConditionalERDynamic

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
    # Lean form (no int64 one-hot, no (1 - x) temporaries): fp = sum(p) - tp, fn = sum(y) - tp over valid voxels.
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


class StructuredConditionalUNetGroupHead(StructuredConditionalUNet):
    """StructuredConditionalUNet plus a coarse group-classification head on the last decoder features."""

    group_head: nn.Conv3d
    append_group_logits_to_main: bool

    def forward(self, x: torch.Tensor, group_ids: Optional[torch.Tensor] = None, return_features: bool = False,
                return_group_logits: bool = False):
        need = return_features or return_group_logits or self.append_group_logits_to_main
        out = self._forward_conditioned(x, group_ids, return_features=need)
        if not need:
            return out
        logits, feature = out["logits"], out["decoder_last"]
        group_logits = self.group_head(feature)
        if self.append_group_logits_to_main and not (return_features or return_group_logits):
            if isinstance(logits, (list, tuple)):
                return [torch.cat((logits[0], group_logits), dim=1), *logits[1:]]
            return torch.cat((logits, group_logits), dim=1)
        pack: Dict[str, object] = {"logits": logits, "group_logits": group_logits}
        if return_features:
            pack["decoder_last"] = feature
        return pack


def attach_group_head(net: StructuredConditionalUNet) -> StructuredConditionalUNetGroupHead:
    channels = int(net.decoder.seg_layers[-1].in_channels)
    net.__class__ = StructuredConditionalUNetGroupHead
    ref = next(net.parameters())
    net.group_head = nn.Conv3d(channels, NUM_GROUP_HEAD_CLASSES, kernel_size=1, bias=True).to(ref.device)
    net.append_group_logits_to_main = str(os.environ.get("NNUNET_STRUCTCOND_APPEND_GROUP_LOGITS", "0")).lower() in {
        "1", "true", "yes", "y"}
    return net


class nnUNetTrainerStructuredConditionalERDynamicGroupHead(nnUNetTrainerStructuredConditionalERDynamic):
    """AnchorSlot ER-dynamic recipe + group-classification head trained on every voxel."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        # explicit signature: nnUNetTrainer records init kwargs by inspecting it
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.lambda_group = float(os.environ.get("NNUNET_STRUCTCOND_LAMBDA_GROUP", "0.5"))
        self.group_lut = build_group_lut()
        self._group_loss_epoch: List[float] = []
        if self.use_text_contrastive or self.use_text_generated_slot_head:
            raise RuntimeError("The group-head trainer does not support the text-conditioning variants.")
        self.print_to_log_file(f"[GroupHead] {NUM_GROUP_HEAD_CLASSES}-class group head, lambda_group={self.lambda_group}")

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels, enable_deep_supervision=True):
        net = nnUNetTrainerStructuredConditionalERDynamic.build_network_architecture(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import, num_input_channels,
            num_output_channels, enable_deep_supervision=enable_deep_supervision)
        return attach_group_head(net)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._to_device_target(batch["target"], self.device)
        target_high = target[0] if isinstance(target, list) else target
        group_ids = self._extract_group_ids_for_batch(batch, target_high)
        bincount = np.bincount(group_ids.detach().cpu().numpy(), minlength=self.num_dynamic_groups)
        self._group_sample_counter_epoch += bincount.astype(np.int64)
        remapped_target, valid_mask, active_slots = self._remap_target_for_group(target, group_ids)

        self.network.zero_grad(set_to_none=True)
        with self._autocast_context():
            pack = self.network(data, group_ids, return_group_logits=True)
        output, group_logits = pack["logits"], pack["group_logits"]
        self._assert_all_finite(output, "network output")
        self._assert_all_finite(group_logits, "group head output")

        loss_output = self._output_to_float(output) if self.loss_in_fp32 else output
        structured_loss = self._compute_structured_loss(loss_output, remapped_target, valid_mask, active_slots)
        valid_high = valid_mask[0] if isinstance(valid_mask, list) else valid_mask
        g_logits = group_logits.float() if self.loss_in_fp32 else group_logits
        group_loss = group_head_loss(g_logits, target_high, valid_high, self.group_lut)
        loss = structured_loss + self.lambda_group * group_loss
        self._assert_all_finite(loss, "training loss")

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self._get_optimizer_parameters(), self.grad_clip_norm)
            if not self._all_finite(grad_norm):
                return self._skip_nonfinite_gradient_update(loss, grad_norm)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self._get_optimizer_parameters(), self.grad_clip_norm)
            if not self._all_finite(grad_norm):
                return self._skip_nonfinite_gradient_update(loss, grad_norm)
            self.optimizer.step()

        self._group_loss_epoch.append(float(group_loss.detach().cpu()))
        zero = structured_loss.detach().cpu().numpy() * 0.0
        # Same key set as the parent step (and its non-finite skip path) so epoch-end collation never diverges.
        return {
            "loss": loss.detach().cpu().numpy(),
            "structured_loss": structured_loss.detach().cpu().numpy(),
            "text_contrast_loss": zero,
            "text_contrast_num_samples": 0,
            "text_contrast_num_labels": 0,
            "text_contrast_sampled_label_ids": [],
            "text_contrast_missing_label_ids": [],
            "text_generated_slot_active_slots": 0,
            "text_generated_slot_inactive_slots": 0,
            "text_generated_slot_logit_mean": 0.0,
            "text_generated_slot_logit_std": 0.0,
            "text_generated_slot_logit_min": 0.0,
            "text_generated_slot_logit_max": 0.0,
        }

    def on_train_epoch_end(self, train_outputs: List[dict]):
        super().on_train_epoch_end(train_outputs)
        if self._group_loss_epoch:
            peak = (f", peak CUDA memory {torch.cuda.max_memory_allocated(self.device) / 2**30:.1f} GiB"
                    if self.device.type == "cuda" else "")
            self.print_to_log_file(f"[GroupHead] epoch {self.current_epoch} mean group loss "
                                   f"{float(np.mean(self._group_loss_epoch)):.4f} (rank-local){peak}")
        self._group_loss_epoch = []


# --- merges from the compact per-group probability dump ----------------------------------------------------

LOW = -1e4


def compact_scores(p: np.ndarray, meta: Sequence[dict], head: Optional[np.ndarray], mode: str = "legacy",
                   weight: float = 1.0, eps: float = 1e-8) -> np.ndarray:
    """32-channel original-label score map from compact probabilities.

    p      C x ...: channel 0 mean background, 1-4 mean anchors, then per group its slot probs and 'other' prob
           (layout described by `meta`, as written by the dump wrapper).
    head   18 x ...: group-head probabilities averaged over the 13 query runs (needed for 'mix' / 'arbiter').
    legacy   the original merge: mean background/anchors, each dynamic label from its own group run.
    mix      geometric mix (1-w) * legacy + w * hierarchical, hierarchical = head(bg/anchor) and
             head(group g) * slot_l / sum of g's slots for dynamic label l.
    arbiter  background/anchor/dynamic decision as legacy; the winning dynamic label re-ranked by
             slot_l * head(group g)^w (w = 0 is legacy).
    """
    S = np.full((32, *p.shape[1:]), LOW, dtype=np.float32)
    S[0] = p[0]
    for c in meta:
        if c["kind"] in ("anchor_mean", "slot"):
            S[int(c["label"])] = p[int(c["index"])]
    if mode == "legacy":
        return S
    if head is None:
        raise ValueError(f"mode {mode!r} needs group-head probabilities")
    slots = [c for c in meta if c["kind"] == "slot"]
    gsum: Dict[int, np.ndarray] = {}
    for c in slots:
        gsum[c["group"]] = gsum.get(c["group"], 0) + p[int(c["index"])]
    dyn = [int(c["label"]) for c in slots]
    if mode == "mix":
        H = np.full_like(S, LOW)
        H[0] = head[0]
        for i, label in enumerate(_ANCHOR_LABELS):
            H[label] = head[1 + i]
        for c in slots:
            H[int(c["label"])] = head[5 + int(c["group"])] * p[int(c["index"])] / np.maximum(gsum[c["group"]], eps)
        used = [0, *_ANCHOR_LABELS, *dyn]
        out = S.copy()
        out[used] = np.exp((1 - weight) * np.log(np.maximum(S[used], eps)) + weight * np.log(np.maximum(H[used], eps)))
        return out
    if mode == "arbiter":
        m = S[dyn].max(axis=0)
        r = np.stack([p[int(c["index"])] * np.power(np.maximum(head[5 + int(c["group"])], eps), weight) for c in slots])
        rmax = np.maximum(r.max(axis=0), 1e-12)
        out = S.copy()
        for i, c in enumerate(slots):
            out[int(c["label"])] = m * r[i] / rmax
        return out
    raise ValueError(f"unknown mode {mode!r}")
