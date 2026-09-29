"""Opt-in Synapse corrections. Historical losses and inference remain unchanged."""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from .dataloader_structured_conditional_synapse import StructuredConditionalDataLoader3D
from .inference_structured_conditional_synapse import reconstruct_original_labels_from_all_groups
from .label_mapping_synapse import (
    COND_SLOT_1_CHANNEL, COND_SLOT_3_CHANNEL, NUM_OUTPUT_CHANNELS, get_group_spec,
)
from .structured_loss_synapse import StructuredConditionalLoss


def active_slots_for_group(group_id: int, batch_size: int, device) -> torch.Tensor:
    slots = torch.arange(3, device=device) < get_group_spec(group_id).num_slots
    return slots.unsqueeze(0).expand(batch_size, -1)


def mask_inactive_slots(logits: torch.Tensor, active_slots: torch.Tensor) -> torch.Tensor:
    """Remove ALL inactive slots from competition, without mutating raw logits.

    Finite minimum rather than -inf keeps zero-loss/invalid-voxel paths finite.
    The loss evaluates FP16/BF16 inputs in FP32, preserving stable reductions.
    Background, anchors and 'other' always remain in the probability denominator.
    """
    if logits.ndim < 3 or logits.shape[1] != NUM_OUTPUT_CHANNELS:
        raise ValueError(f"Expected [B, {NUM_OUTPUT_CHANNELS}, ...], got {tuple(logits.shape)}")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    if tuple(active_slots.shape) != (logits.shape[0], 3):
        raise ValueError("active_slots must have shape [B, 3]")
    active = torch.ones(logits.shape[:2], device=logits.device, dtype=torch.bool)
    active[:, COND_SLOT_1_CHANNEL:COND_SLOT_3_CHANNEL + 1] = active_slots.to(logits.device).bool()
    active = active.reshape(*active.shape, *([1] * (logits.ndim - 2)))
    return logits.masked_fill(~active, torch.finfo(logits.dtype).min)


def mask_group_logits(logits: torch.Tensor, group_id: int) -> torch.Tensor:
    return mask_inactive_slots(logits, active_slots_for_group(group_id, logits.shape[0], logits.device))


@torch.no_grad()
def reconstruct_consistent(logits_by_group, fixed_merge_mode="mean"):
    return reconstruct_original_labels_from_all_groups(
        {g: mask_group_logits(x, g) for g, x in logits_by_group.items()}, fixed_merge_mode
    )


class ConsistentSynapseLoss(StructuredConditionalLoss):
    """Identical component weights, with one mask before every loss component.

    The optional ablation averages individual sample Dice ratios per conditional
    slot rather than pooling different organs' voxel statistics across samples.
    Fixed anchors and 'other' keep batch Dice; slot weighting stays unchanged.
    This is sample-balanced conditional Dice, not an original-organ batch loss.
    """

    def __init__(self, config=None, conditional_sample_dice=False):
        super().__init__(config)
        self.conditional_sample_dice = bool(conditional_sample_dice)

    def _mask_unused_cond3_channel_for_ce(self, logits, active_conditional_slots):
        # All inactive slots have already been masked in forward. Do not replace
        # the mask by the legacy -1e4 constant for just the third slot.
        return logits

    def forward(self, logits, target, valid_mask, active_conditional_slots, return_components=False):
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        logits = mask_inactive_slots(logits, active_conditional_slots)
        return super().forward(logits, target, valid_mask, active_conditional_slots, return_components)

    def _masked_soft_dice(self, logits, target, valid_mask, class_mask):
        if not self.conditional_sample_dice or not self.config.batch_dice:
            return super()._masked_soft_dice(logits, target, valid_mask, class_mask)
        probs = torch.softmax(logits, dim=1)
        labels = target[:, 0].long().clamp(0, NUM_OUTPUT_CHANNELS - 1)
        onehot = F.one_hot(labels, NUM_OUTPUT_CHANNELS).movedim(-1, 1).to(probs.dtype)
        valid = valid_mask.to(probs.dtype)
        mask = self._expand_per_class_mask(class_mask.to(probs.dtype), logits.ndim)
        probs, onehot = probs * mask, onehot * mask
        axes = tuple(range(2, logits.ndim))
        inter = (probs * onehot * valid).sum(axes)
        denom = ((probs + onehot) * valid).sum(axes)
        sample_dice = (2 * inter + self.config.smooth) / (denom + self.config.smooth).clamp_min(1e-8)
        # Entirely invalid samples must not contribute a spurious perfect Dice.
        has_valid = valid.sum(axes)[:, 0] > 0
        entries = class_mask & has_valid[:, None]
        sample_sum = (sample_dice * entries).sum(0)
        counts = entries.sum(0).to(probs.dtype)
        inter, denom = inter.sum(0), denom.sum(0)
        if self.config.ddp:
            if not dist.is_initialized():
                raise RuntimeError("DDP loss requires an initialized process group")
            inter = AllGatherGrad.apply(inter).sum(0)
            denom = AllGatherGrad.apply(denom).sum(0)
            sample_sum = AllGatherGrad.apply(sample_sum).sum(0)
            dist.all_reduce(counts)
        batch_dice = (2 * inter + self.config.smooth) / (denom + self.config.smooth).clamp_min(1e-8)
        conditional = torch.zeros(NUM_OUTPUT_CHANNELS, device=logits.device, dtype=torch.bool)
        conditional[COND_SLOT_1_CHANNEL:COND_SLOT_3_CHANNEL + 1] = True
        dice = torch.where(conditional, sample_sum / counts.clamp_min(1), batch_dice)
        selected = counts > 0
        if not torch.any(selected):
            return self._safe_zero(logits)
        return 1 - dice[selected].mean()


class QueryAwareSynapseDataLoader(StructuredConditionalDataLoader3D):
    """Align only the existing forced-foreground subset; preserve random crops."""

    def get_bbox_for_group(self, shape, force_fg, class_locations, group_id):
        eligible = [label for label in get_group_spec(group_id).original_labels
                    if label in class_locations and len(class_locations[label]) > 0]
        if force_fg and eligible:
            # Uniform over eligible organs, not proportional to organ volume.
            label = int(self.rng.choice(eligible))
            return self.get_bbox(shape, force_fg, class_locations, overwrite_class=label)
        return super().get_bbox_for_group(shape, force_fg, class_locations, group_id)
