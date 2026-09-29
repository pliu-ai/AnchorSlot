"""Differentiable reconstruction supervision; inference and output head unchanged."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .label_mapping_synapse import (
    FIXED_OUTPUT_TO_ORIGINAL, NUM_DYNAMIC_GROUPS, NUM_ORIGINAL_LABELS,
    get_conditional_channel_to_original_label,
)
from .synapse_consistency import mask_group_logits


def masked_probabilities(logits, group):
    # Match corrected inference: mask before softmax, never supervise dead slots.
    return mask_group_logits(logits.float(), group).softmax(1)


def reconstruct_scores(probabilities):
    """Same label mapping and mean aggregation as inference, WITH autograd.

    Values are confidence scores, not yet a categorical distribution. All five
    groups are required. Do not call the historical @no_grad reconstruction here.
    """
    if set(probabilities) != set(range(NUM_DYNAMIC_GROUPS)):
        raise ValueError("Reconstruction requires exactly all five query groups")
    shape = probabilities[0].shape
    if len(shape) < 3 or shape[1] != 9:
        raise ValueError("Expected [B, 9, ...] probabilities")
    if any(p.shape != shape for p in probabilities.values()):
        raise ValueError("Query tensor shapes differ")
    ordered = [probabilities[g] for g in range(NUM_DYNAMIC_GROUPS)]
    channels = {0: torch.stack([p[:, 0] for p in ordered]).mean(0)}
    for channel, label in FIXED_OUTPUT_TO_ORIGINAL.items():
        channels[label] = torch.stack([p[:, channel] for p in ordered]).mean(0)
    for g, p in enumerate(ordered):
        for channel, label in get_conditional_channel_to_original_label(g).items():
            if label in channels:
                raise ValueError("Original label assigned to more than one output")
            channels[label] = p[:, channel]
    if set(channels) != set(range(NUM_ORIGINAL_LABELS)):
        raise ValueError("Incomplete original-label reconstruction")
    return torch.stack([channels[c] for c in range(NUM_ORIGINAL_LABELS)], dim=1)


def original_label_loss(scores, target, smooth=1e-5):
    """NLL + foreground batch Dice of normalized reconstructed probabilities.

    A second softmax on probabilities would change their meaning. Instead divide
    by their positive sum; this preserves per-voxel argmax. This is patch-level
    auxiliary supervision, NOT a change to sliding-window/native inference.
    Padding labels <0 are excluded from every reduction.
    """
    if scores.shape[1] != NUM_ORIGINAL_LABELS:
        raise ValueError("Expected 14 original-label score channels")
    if target.shape != (scores.shape[0], 1, *scores.shape[2:]):
        raise ValueError("Target geometry does not match reconstructed scores")
    labels = target[:, 0].long()
    valid = labels >= 0
    if torch.any(labels[valid] >= NUM_ORIGINAL_LABELS):
        raise ValueError("Unknown original label")
    if not torch.any(valid):
        return scores.sum() * 0.0
    scores = scores.float()
    p = scores / scores.sum(1, keepdim=True).clamp_min(torch.finfo(scores.dtype).tiny)
    safe_labels = labels.clamp(0, NUM_ORIGINAL_LABELS - 1)
    nll = -p.gather(1, safe_labels[:, None])[:, 0].clamp_min(1e-8).log()[valid].mean()
    onehot = F.one_hot(safe_labels, NUM_ORIGINAL_LABELS).movedim(-1, 1).to(p.dtype)
    mask = valid[:, None].to(p.dtype)
    axes = (0, *range(2, p.ndim))
    intersection = (p * onehot * mask).sum(axes)
    denominator = ((p + onehot) * mask).sum(axes)
    dice = (2 * intersection + smooth) / (denominator + smooth)
    return nll + 1 - dice[1:].mean()
