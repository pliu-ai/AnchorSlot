from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nnunetv2.utilities.ddp_allgather import AllGatherGrad

from .label_mapping_er_dynamic import (
    BACKGROUND_CHANNEL,
    COND_SLOT_1_CHANNEL,
    COND_SLOT_2_CHANNEL,
    COND_SLOT_3_CHANNEL,
    ECS_CHANNEL,
    NUCPL_CHANNEL,
    NUM_OUTPUT_CHANNELS,
    OTHER_CHANNEL,
)

SUPPRESSION_MODES = ("pooled", "region_balanced")


def mask_inactive_conditional_logits(
    logits: torch.Tensor, active_conditional_slots: torch.Tensor
) -> torch.Tensor:
    """Remove unused slots from the distribution shared by loss and inference.

    Keep background, fixed anchors and ``other`` in the competition. A finite
    FP32 sentinel also keeps masked reductions and empty losses well defined.
    """
    if logits.ndim < 3 or logits.shape[1] != NUM_OUTPUT_CHANNELS:
        raise ValueError(f"Expected logits [B, {NUM_OUTPUT_CHANNELS}, ...]")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    active = torch.as_tensor(active_conditional_slots, device=logits.device, dtype=torch.bool)
    if active.shape != (logits.shape[0], 3):
        raise ValueError("active_conditional_slots must have shape [B, 3]")
    channels = torch.ones((logits.shape[0], NUM_OUTPUT_CHANNELS), device=logits.device, dtype=torch.bool)
    channels[:, COND_SLOT_1_CHANNEL : COND_SLOT_3_CHANNEL + 1] = active
    expanded = channels.view(*channels.shape, *([1] * (logits.ndim - 2)))
    stable_logits = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    return stable_logits.masked_fill(~expanded, torch.finfo(stable_logits.dtype).min)


@dataclass
class StructuredLossConfig:
    """Configuration for structured conditional loss components."""

    lambda_ce: float = 1.0
    lambda_dice: float = 1.0
    lambda_cond: float = 0.25
    lambda_suppress: float = 0.1

    enable_conditional_focus: bool = True
    enable_suppression: bool = True

    batch_dice: bool = True
    smooth: float = 1e-5
    ddp: bool = False

    class_balance_mode: str = "none"
    class_balance_power: float = 0.5
    class_balance_min_weight: float = 0.25
    class_balance_max_weight: float = 4.0
    class_balance_eps: float = 1e-6

    # Drop Dice entries whose ground truth is empty (per sample, or per batch with batch_dice),
    # instead of averaging in their near-constant ~1 loss. Off reproduces the original loss.
    dice_exclude_empty_gt: bool = False
    # 'pooled': mean over all non-slot voxels (original, background-dominated);
    # 'region_balanced': mean of the background, anchor and non-query-foreground region means.
    suppression_mode: str = "pooled"


class StructuredConditionalLoss(nn.Module):
    """
    Structured loss for fixed 9-channel output with dynamic conditional slots.

    Supports:
    - masked CE
    - masked Soft Dice over active foreground channels
    - optional conditional-slot focus loss
    - optional conditional suppression loss
    """

    def __init__(self, config: Optional[StructuredLossConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else StructuredLossConfig()
        if str(self.config.suppression_mode) not in SUPPRESSION_MODES:
            raise ValueError(
                f"StructuredLossConfig.suppression_mode must be one of {SUPPRESSION_MODES}, "
                f"got {self.config.suppression_mode!r}"
            )

    @staticmethod
    def _expand_per_class_mask(class_mask: torch.Tensor, ndim: int) -> torch.Tensor:
        view_shape = [class_mask.shape[0], class_mask.shape[1]] + [1] * (ndim - 2)
        return class_mask.view(*view_shape)

    def _mask_unused_cond3_channel_for_ce(self, logits: torch.Tensor, active_conditional_slots: torch.Tensor) -> torch.Tensor:
        """Compatibility hook for DDW subclasses; mask every inactive slot."""
        return mask_inactive_conditional_logits(logits, active_conditional_slots)

    @staticmethod
    def _build_foreground_class_mask(active_conditional_slots: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Foreground mask for Dice: fixed classes + active cond slots + other."""
        b = int(active_conditional_slots.shape[0])
        class_mask = torch.zeros((b, NUM_OUTPUT_CHANNELS), dtype=torch.bool, device=device)

        # Fixed foreground channels.
        class_mask[:, 1:5] = True

        # Conditional slots.
        class_mask[:, COND_SLOT_1_CHANNEL] = active_conditional_slots[:, 0]
        class_mask[:, COND_SLOT_2_CHANNEL] = active_conditional_slots[:, 1]
        class_mask[:, COND_SLOT_3_CHANNEL] = active_conditional_slots[:, 2]

        # Other channel is part of supervision.
        class_mask[:, OTHER_CHANNEL] = True
        return class_mask

    @staticmethod
    def _build_conditional_class_mask(active_conditional_slots: torch.Tensor, device: torch.device) -> torch.Tensor:
        b = int(active_conditional_slots.shape[0])
        class_mask = torch.zeros((b, NUM_OUTPUT_CHANNELS), dtype=torch.bool, device=device)
        class_mask[:, COND_SLOT_1_CHANNEL] = active_conditional_slots[:, 0]
        class_mask[:, COND_SLOT_2_CHANNEL] = active_conditional_slots[:, 1]
        class_mask[:, COND_SLOT_3_CHANNEL] = active_conditional_slots[:, 2]
        return class_mask

    @staticmethod
    def _safe_zero(logits: torch.Tensor) -> torch.Tensor:
        return logits.reshape(-1)[:0].float().sum()

    def _masked_ce(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        active_conditional_slots: torch.Tensor,
    ) -> torch.Tensor:
        logits_ce = self._mask_unused_cond3_channel_for_ce(logits, active_conditional_slots)
        target_labels = target[:, 0].long().clamp(min=0, max=NUM_OUTPUT_CHANNELS - 1)
        valid = valid_mask[:, 0].float()

        denom = valid.sum()
        if float(denom.item()) <= 0.0:
            return self._safe_zero(logits)

        ce_weight = self._build_ce_class_weights(
            target_labels=target_labels,
            valid=valid,
            active_conditional_slots=active_conditional_slots,
            device=logits.device,
        )
        ce_map = F.cross_entropy(logits_ce, target_labels, weight=ce_weight, reduction="none")
        return (ce_map * valid).sum() / denom.clamp_min(1.0)

    def _build_ce_class_weights(
        self,
        target_labels: torch.Tensor,
        valid: torch.Tensor,
        active_conditional_slots: torch.Tensor,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        mode = str(self.config.class_balance_mode).strip().lower()
        if mode in {"", "none", "off", "0", "false"}:
            return None
        if mode in {"remap", "remap_sqrt", "structured", "structured_sqrt"}:
            return self._build_remapped_ce_weights(target_labels, valid, device)
        if mode in {"slot", "slot_sqrt", "active_slot", "active_slots"}:
            return self._build_slot_ce_weights(target_labels, valid, active_conditional_slots, device)
        raise ValueError(
            "StructuredLossConfig.class_balance_mode must be one of: "
            "none, remap_sqrt, slot_sqrt"
        )

    def _inverse_count_weights(self, counts: torch.Tensor, present_mask: torch.Tensor) -> torch.Tensor:
        weights = torch.ones_like(counts)
        if not torch.any(present_mask):
            return weights

        counts_present = counts[present_mask].float().clamp_min(float(self.config.class_balance_eps))
        mean_count = counts_present.mean().clamp_min(float(self.config.class_balance_eps))
        balanced = (mean_count / counts_present).pow(float(self.config.class_balance_power))
        balanced = balanced.clamp(
            min=float(self.config.class_balance_min_weight),
            max=float(self.config.class_balance_max_weight),
        )
        balanced = balanced / balanced.mean().clamp_min(float(self.config.class_balance_eps))
        weights[present_mask] = balanced
        return weights

    def _build_remapped_ce_weights(
        self,
        target_labels: torch.Tensor,
        valid: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        labels = target_labels[valid > 0].reshape(-1)
        if labels.numel() == 0:
            return torch.ones((NUM_OUTPUT_CHANNELS,), dtype=torch.float32, device=device)
        counts = torch.bincount(labels, minlength=NUM_OUTPUT_CHANNELS).float().to(device)
        counts = counts[:NUM_OUTPUT_CHANNELS]
        return self._inverse_count_weights(counts, counts > 0)

    def _build_slot_ce_weights(
        self,
        target_labels: torch.Tensor,
        valid: torch.Tensor,
        active_conditional_slots: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        labels = target_labels[valid > 0].reshape(-1)
        weights = torch.ones((NUM_OUTPUT_CHANNELS,), dtype=torch.float32, device=device)
        if labels.numel() == 0:
            return weights

        counts = torch.bincount(labels, minlength=NUM_OUTPUT_CHANNELS).float().to(device)
        counts = counts[:NUM_OUTPUT_CHANNELS]
        slot_channels = torch.as_tensor(
            [COND_SLOT_1_CHANNEL, COND_SLOT_2_CHANNEL, COND_SLOT_3_CHANNEL],
            dtype=torch.long,
            device=device,
        )
        active_slots = active_conditional_slots.to(device=device, dtype=torch.bool).any(dim=0)
        slot_counts = counts[slot_channels]
        present_slots = active_slots & (slot_counts > 0)
        if not torch.any(present_slots):
            return weights

        balanced_slots = self._inverse_count_weights(slot_counts, present_slots)
        weights[slot_channels] = balanced_slots
        return weights

    def _masked_soft_dice(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        class_mask: torch.Tensor,
    ) -> torch.Tensor:
        # ``forward`` and ``_conditional_focus_loss`` supply logits with all
        # inactive slots removed. class_mask selects Dice entries only; it
        # must not remove the competing background/anchor/other classes.
        probs = torch.softmax(logits, dim=1)
        target_labels = target[:, 0].long().clamp(min=0, max=NUM_OUTPUT_CHANNELS - 1)
        target_onehot = F.one_hot(target_labels, num_classes=NUM_OUTPUT_CHANNELS)
        target_onehot = target_onehot.permute(0, -1, *range(1, target_onehot.ndim - 1)).float()

        valid = valid_mask.float()
        class_mask_float = self._expand_per_class_mask(class_mask.float(), logits.ndim)

        probs = probs * class_mask_float
        target_onehot = target_onehot * class_mask_float

        axes = tuple(range(2, logits.ndim))
        intersect = (probs * target_onehot * valid).sum(dim=axes)
        pred_sum = (probs * valid).sum(dim=axes)
        gt_sum = (target_onehot * valid).sum(dim=axes)

        if self.config.batch_dice:
            if self.config.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(dim=0)
                pred_sum = AllGatherGrad.apply(pred_sum).sum(dim=0)
                gt_sum = AllGatherGrad.apply(gt_sum).sum(dim=0)

            intersect = intersect.sum(dim=0)
            pred_sum = pred_sum.sum(dim=0)
            gt_sum = gt_sum.sum(dim=0)

            dice = (2.0 * intersect + self.config.smooth) / (pred_sum + gt_sum + self.config.smooth).clamp_min(1e-8)
            valid_classes = class_mask.any(dim=0).to(torch.int64)

            if self.config.ddp and dist.is_available() and dist.is_initialized():
                dist.all_reduce(valid_classes, op=dist.ReduceOp.MAX)
            valid_classes = valid_classes.bool()
            if self.config.dice_exclude_empty_gt:
                # gt_sum is already summed over the (DDP-gathered) batch, so every rank agrees.
                valid_classes = valid_classes & (gt_sum.detach() > 0)

            if not torch.any(valid_classes):
                return self._safe_zero(logits)
            return 1.0 - dice[valid_classes].mean()

        dice = (2.0 * intersect + self.config.smooth) / (pred_sum + gt_sum + self.config.smooth).clamp_min(1e-8)
        valid_entries = class_mask
        if self.config.dice_exclude_empty_gt:
            valid_entries = valid_entries & (gt_sum.detach() > 0)
        if not torch.any(valid_entries):
            return self._safe_zero(logits)
        return 1.0 - dice[valid_entries].mean()

    def _conditional_focus_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        active_conditional_slots: torch.Tensor,
    ) -> torch.Tensor:
        logits = mask_inactive_conditional_logits(logits, active_conditional_slots)
        cond_class_mask = self._build_conditional_class_mask(active_conditional_slots, logits.device)
        if not torch.any(cond_class_mask):
            return self._safe_zero(logits)
        return self._masked_soft_dice(logits, target, valid_mask, cond_class_mask)

    def _suppression_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        active_conditional_slots: torch.Tensor,
    ) -> torch.Tensor:
        """
        Suppress active conditional slots in non-conditional target regions.
        """
        logits = mask_inactive_conditional_logits(logits, active_conditional_slots)
        probs = torch.softmax(logits, dim=1)
        cond_probs = probs[:, COND_SLOT_1_CHANNEL : COND_SLOT_3_CHANNEL + 1]

        active = active_conditional_slots.float()
        active = active.view(active.shape[0], active.shape[1], *([1] * (logits.ndim - 2)))

        cond_region = ((target >= COND_SLOT_1_CHANNEL) & (target <= COND_SLOT_3_CHANNEL)).float()
        suppression_region = valid_mask.float() * (1.0 - cond_region)

        if self.config.suppression_mode == "region_balanced":
            return self._region_balanced_suppression(cond_probs, active, target, valid_mask, logits)

        numer = (cond_probs * active * suppression_region).sum()
        denom = (active * suppression_region).sum().clamp_min(1.0)
        return numer / denom

    def _region_balanced_suppression(
        self,
        cond_probs: torch.Tensor,
        active: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean over the background, fixed-anchor and non-query-foreground regions of the
        per-region mean active-slot probability, so rare hard negatives (anchors, other
        organelles) are not outweighed by background voxels. Empty regions are skipped.
        """
        del logits
        valid = valid_mask.float()
        regions = (
            target == BACKGROUND_CHANNEL,
            (target >= ECS_CHANNEL) & (target <= NUCPL_CHANNEL),
            target == OTHER_CHANNEL,
        )
        numers, denoms = [], []
        for region in regions:
            weight = active * (region.float() * valid)
            numers.append((cond_probs * weight).sum())
            denoms.append(weight.sum())
        numer = torch.stack(numers)
        denom = torch.stack(denoms)
        present = (denom > 0).to(numer.dtype)
        # Sync-free mean over non-empty regions; all-empty gives a graph-connected zero.
        return (present * numer / denom.clamp_min(1.0)).sum() / present.sum().clamp_min(1.0)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        active_conditional_slots: torch.Tensor,
        return_components: bool = False,
    ):
        # All objectives must see the same distribution, including singleton
        # groups where both slot 2 and slot 3 are unused.
        logits = mask_inactive_conditional_logits(logits, active_conditional_slots)
        ce_loss = self._masked_ce(logits, target, valid_mask, active_conditional_slots)

        foreground_class_mask = self._build_foreground_class_mask(active_conditional_slots, logits.device)
        dice_loss = self._masked_soft_dice(logits, target, valid_mask, foreground_class_mask)

        cond_loss = self._safe_zero(logits)
        if self.config.enable_conditional_focus and self.config.lambda_cond > 0:
            cond_loss = self._conditional_focus_loss(logits, target, valid_mask, active_conditional_slots)

        suppress_loss = self._safe_zero(logits)
        if self.config.enable_suppression and self.config.lambda_suppress > 0:
            suppress_loss = self._suppression_loss(logits, target, valid_mask, active_conditional_slots)

        total = (
            self.config.lambda_ce * ce_loss
            + self.config.lambda_dice * dice_loss
            + self.config.lambda_cond * cond_loss
            + self.config.lambda_suppress * suppress_loss
        )

        if not return_components:
            return total

        components: Dict[str, torch.Tensor] = {
            "total": total,
            "ce": ce_loss,
            "dice": dice_loss,
            "cond_focus": cond_loss,
            "suppress": suppress_loss,
        }
        return total, components
