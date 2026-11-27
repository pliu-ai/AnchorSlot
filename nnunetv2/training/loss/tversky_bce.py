"""Tversky + BCE loss on per-label sigmoid outputs, with ignored (unannotated) target channels.

Used by the MISO-style flat baseline (Kuruba et al., bioRxiv 2026: Tversky alpha=0.3, beta=0.7, plus BCE).
Targets are region one-hot maps (B, C, ...); a value of -1 marks voxels to ignore, which the trainer uses for
labels that a crop does not annotate. Written to keep few full-size temporaries: 31 channels at 160^3 are ~0.5 GB
per sample in float32.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MaskedTverskyBCELoss(nn.Module):
    """BCE (mean over valid voxels) + (1 - mean Tversky index over valid sample-channel pairs).

    alpha weights false positives and beta false negatives; beta > alpha favours recall of small structures.
    The Tversky index is computed per sample and channel (no batch pooling), matching batch_dice=False.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-5):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape != target.shape:
            raise ValueError(f"logits {tuple(logits.shape)} and target {tuple(target.shape)} must match")
        logits = logits.float()
        axes = tuple(range(2, logits.ndim))
        masked = bool((target < 0).any())
        if masked:
            valid = target >= 0
            positive = (target > 0).float()
            weight = valid.float()
            bce = F.binary_cross_entropy_with_logits(logits, positive, weight=weight, reduction="sum")
            bce = bce / weight.sum().clamp_min(1.0)
            prob = torch.sigmoid(logits) * weight
            channel_valid = valid.flatten(2).any(-1)
            del weight
        else:
            positive = target.float()
            bce = F.binary_cross_entropy_with_logits(logits, positive)
            prob = torch.sigmoid(logits)
            channel_valid = None

        # positive is zero wherever the target is ignored, so tp and fn only count valid voxels; prob is zeroed there.
        tp = (prob * positive).sum(axes)
        fp = prob.sum(axes) - tp
        fn = positive.sum(axes) - tp
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        if channel_valid is None:
            return bce + (1.0 - tversky.mean())
        if not torch.any(channel_valid):
            return bce
        return bce + (1.0 - tversky[channel_valid].mean())
