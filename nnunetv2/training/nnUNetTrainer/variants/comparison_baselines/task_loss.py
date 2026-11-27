"""DoDNet binary task supervision with explicit inactive-slot and ignore masks."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .dodnet import DEFAULT_DODNET_TASK_LABELS


def binary_task_loss(logits, target, task_ids, ignore_label=None, smooth=1e-5):
    if target.ndim == logits.ndim:
        if target.shape[1] != 1:
            raise ValueError("Expected one semantic target channel")
        target = target[:, 0]
    if tuple(target.shape) != (logits.shape[0], *logits.shape[2:]):
        raise ValueError("Target and logits spatial shapes differ")
    if logits.shape[1] != 3:
        raise ValueError("DoDNet CellMap tasks require three binary channels")
    task_ids = torch.as_tensor(task_ids, device=logits.device)
    if task_ids.dtype == torch.bool or task_ids.is_floating_point() or task_ids.is_complex():
        raise ValueError("Task IDs must be integers")
    task_ids = task_ids.long().reshape(-1)
    if task_ids.numel() != logits.shape[0] or torch.any((task_ids < 0) | (task_ids >= 17)):
        raise ValueError("One valid task ID in [0,17) is required per sample")
    valid = target >= 0
    if ignore_label is not None:
        valid &= target != ignore_label
    if torch.any(valid & (target > 31)):
        raise ValueError("Expected original CellMap labels 0..31")
    if target.is_floating_point() and torch.any(valid & (target != target.floor())):
        raise ValueError("Target labels must be integers")
    table = torch.full((17, 3), -1, device=logits.device, dtype=torch.long)
    for i, labels in enumerate(DEFAULT_DODNET_TASK_LABELS):
        table[i, :len(labels)] = torch.as_tensor(labels, device=logits.device)
    selected = table[task_ids]
    active = selected >= 0
    labels = selected.reshape(*selected.shape, *([1] * (logits.ndim - 2)))
    binary = (target[:, None] == labels).float()
    mask = valid[:, None] & active.reshape(*active.shape, *([1] * (logits.ndim - 2)))
    logits = logits.float()
    ce = (F.binary_cross_entropy_with_logits(logits, binary, reduction="none") * mask).sum()
    ce = ce / mask.sum().clamp_min(1)
    probs = logits.sigmoid() * mask
    binary = binary * mask
    axes = tuple(range(2, logits.ndim))
    dice = (2 * (probs * binary).sum(axes) + smooth) / (probs.sum(axes) + binary.sum(axes) + smooth)
    eligible = active & valid.flatten(1).any(1)[:, None]
    return ce - (dice * eligible).sum() / eligible.sum().clamp_min(1)


def sample_task_ids(target, rng, p_present=.8):
    """Choose a present task with p=.8, otherwise an absent task, after cropping.

    Spatial patches use the same nnU-Net sampler as the other baselines. Task
    selection alone is conditional; every foreground class has a dynamic head.
    """
    import numpy as np
    target = target.detach().cpu()
    tasks = []
    for sample in target:
        labels = set(int(x) for x in torch.unique(sample).tolist())
        present = [i for i, task in enumerate(DEFAULT_DODNET_TASK_LABELS) if labels.intersection(task)]
        absent = [i for i in range(len(DEFAULT_DODNET_TASK_LABELS)) if i not in present]
        if not present:
            candidates = list(range(len(DEFAULT_DODNET_TASK_LABELS)))
        elif rng.random() < p_present or not absent:
            candidates = present
        else:
            candidates = absent
        tasks.append(int(rng.choice(candidates)))
    return np.asarray(tasks, dtype=np.int64)
