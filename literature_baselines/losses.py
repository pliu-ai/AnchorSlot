import torch
from torch.nn import functional as F
from .models import TASKS


def valid_voxels(target, annotated):
    """Unknown background on partially annotated crops is conservatively ignored."""
    y = target[:, 0].long()
    valid = (y >= 0) & (y < 32)
    known = annotated.gather(1, y.clamp(0, 31).flatten(1)).reshape(y.shape)
    return valid & known


def remap_task(target, query, annotated):
    y = target[:, 0].long()
    remapped = torch.zeros_like(y)
    active = torch.zeros((len(query), 5), device=y.device, dtype=torch.bool)
    for b, q in enumerate(query.tolist()):
        labels = TASKS[q]
        active[b, :len(labels) + 1] = True
        for c, original in enumerate(labels, 1):
            remapped[b][y[b] == original] = c
    return remapped, valid_voxels(target, annotated), active


def task_loss(logits, target, query, annotated):
    y, valid, active = remap_task(target, query, annotated)
    z = logits.float().masked_fill(~active[:, :, None, None, None], -1e4)
    ce = F.cross_entropy(z, y, reduction='none')
    ce = (ce * valid).sum() / valid.sum().clamp_min(1)
    p = z.softmax(1)
    scores = []
    for b in range(len(query)):
        if valid[b].any():
            for c in range(1, int(active[b].sum())):
                positive = (y[b] == c) & valid[b]
                prob = p[b, c] * valid[b]
                scores.append((2 * (prob * positive).sum() + 1e-5) /
                              (prob.sum() + positive.sum() + 1e-5))
    dice = 1 - torch.stack(scores).mean() if scores else z.sum() * 0
    return ce + dice


def binary_loss(logits, target, query, annotated):
    valid = valid_voxels(target, annotated)
    positive = (target[:, 0] == (query + 1)[:, None, None, None]).float()
    z = logits[:, 0].float()
    bce = (F.binary_cross_entropy_with_logits(z, positive, reduction='none') * valid).sum()
    bce = bce / valid.sum().clamp_min(1)
    p = z.sigmoid() * valid
    positive = positive * valid
    dims = (1, 2, 3)
    score = (2 * (p * positive).sum(dims) + 1e-5) / (p.sum(dims) + positive.sum(dims) + 1e-5)
    sample_valid = valid.flatten(1).any(1)
    dice = 1 - score[sample_valid].mean() if sample_valid.any() else z.sum() * 0
    return bce + dice


def miso_loss(logits, target, annotated, alpha=0.3, beta=0.7, smooth=1e-5):
    """32-channel masked sigmoid BCE + mean per-sample/channel Tversky.

Chunking avoids a full 32-channel float32 one-hot target at 160^3.
Includes background, unlike the earlier 31-sigmoid experiment.
"""
    valid = valid_voxels(target, annotated)
    axes = (2, 3, 4)
    bce_sum = logits.new_zeros((), dtype=torch.float32)
    valid_count = logits.new_zeros((), dtype=torch.float32)
    score_sum = logits.new_zeros((), dtype=torch.float32)
    score_count = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, 32, 4):
        z = logits[:, start:start + 4].float()
        labels = torch.arange(start, start + 4, device=z.device)[None, :, None, None, None]
        positive = (target == labels).float()
        mask = valid[:, None] & annotated[:, start:start + 4, None, None, None]
        bce_sum = bce_sum + (F.binary_cross_entropy_with_logits(z, positive, reduction='none') * mask).sum()
        valid_count = valid_count + mask.sum()
        p = z.sigmoid() * mask
        positive = positive * mask
        tp = (p * positive).sum(axes)
        fp = p.sum(axes) - tp
        fn = positive.sum(axes) - tp
        score = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        channel_valid = mask.flatten(2).any(2)
        score_sum = score_sum + (score * channel_valid).sum()
        score_count = score_count + channel_valid.sum()
    overlap = torch.where(score_count > 0, 1 - score_sum / score_count.clamp_min(1), score_sum * 0)
    return bce_sum / valid_count.clamp_min(1) + overlap


def deep_loss(method, outputs, targets, query, annotated):
    if not isinstance(outputs, (list, tuple)):
        outputs, targets = [outputs], [targets]
    weights = [1 / 2**i for i in range(len(outputs))]
    if len(weights) > 1:
        weights[-1] = 1e-6  # keep all auxiliary heads in the DDP graph
    total = sum(weights)
    losses = []
    for o, t, w in zip(outputs, targets, weights):
        if method == 'uniseg':
            l = task_loss(o, t, query, annotated)
        elif method == 'conemos':
            l = binary_loss(o, t, query, annotated)
        else:
            l = miso_loss(o, t, annotated)
        losses.append(l * (w / total))
    return sum(losses)
