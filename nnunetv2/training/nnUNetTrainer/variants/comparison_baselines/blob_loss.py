"""Flat semantic CE + nnU-Net Dice + component-balanced blob Dice.

References: Kofler et al., https://arxiv.org/abs/2205.08209 (Eq. 1--3),
https://github.com/neuronflow/blob_loss, and the one-vs-rest/class aggregation
in Kundu et al., https://arxiv.org/abs/2604.24276 (Sec. 2.1--2.4). This uses
Dice ONLY inside each blob, not the latter paper's local Dice+BCE objective.

For component k of class c, let t_k=sum_{i in k} p_c(i), n_k=|k|, and
b_c=sum_{i valid, y_i != c} p_c(i). Its loss is
    1 - (2*t_k + smooth) / (t_k + n_k + b_c + smooth).
All other components OF THE SAME CLASS are excluded from this denominator.
All valid one-vs-rest background, including other semantic classes, remains.
Components are averaged within present classes, then classes within each image,
then images containing foreground. Empty images contribute no blob term; the
global CE/Dice still supervises their false positives. No background components
are generated, no component-size truncation is used, and no [K, *spatial] masks
are allocated. Scalar component sums are computed with scatter_add.

Without instance_ids, blobs are semantic connected components, NOT biological
instances (touching same-class objects remain merged). Compute these after the
last spatial augmentation/crop and separately for each supervised resolution.
prepare_blob_components can run on the augmented CPU target in a data worker,
avoiding GPU-to-CPU synchronization in the loss. Alternatively, aligned positive
instance IDs can preserve original instance identity through crops; disconnected
fragments with the same ID remain one instance. Precomputed IDs must undergo the
same crop/nearest-neighbor transforms as the target. Never reuse a component map
from before a shape-changing augmentation or from another deep-supervision scale.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage
import torch
from torch import nn
import torch.nn.functional as F

from nnunetv2.utilities.ddp_allgather import AllGatherGrad

try:
    import cc3d
except ImportError:  # scipy is already an nnU-Net dependency; cc3d is optional.
    cc3d = None


@dataclass(frozen=True)
class BlobComponents:
    """Augmented-target component map; zero is background/ignored, IDs are global.

    Metadata entry j describes label j+1. Objects returned by
    prepare_blob_components live on CPU and can be reused only with the exact
    same semantic target, validity mask, and spatial resolution.
    """

    labels: torch.Tensor
    class_ids: torch.Tensor
    batch_ids: torch.Tensor
    sizes: torch.Tensor
    num_classes: int


def _without_channel(value: torch.Tensor, shape: tuple[int, ...], name: str) -> torch.Tensor:
    value = torch.as_tensor(value)
    if tuple(value.shape) == (shape[0], 1, *shape[1:]):
        value = value[:, 0]
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape} or {(shape[0], 1, *shape[1:])}; got {tuple(value.shape)}")
    return value.detach()


def _semantic_and_valid(target, shape, num_classes, ignore_label, valid_mask):
    raw = _without_channel(target, shape, "target")
    valid = torch.ones(shape, device=raw.device, dtype=torch.bool)
    if valid_mask is not None:
        valid &= _without_channel(valid_mask, shape, "valid_mask").to(device=raw.device, dtype=torch.bool)
    if raw.is_floating_point():
        if torch.any(valid & (~torch.isfinite(raw) | (raw != raw.round()))):
            raise ValueError("Semantic targets must contain finite integer labels.")
    labels = raw.long()
    valid &= labels >= 0  # Padding/ignore values -1, -100, etc. are never classes.
    if ignore_label is not None:
        valid &= labels != ignore_label
    if torch.any(valid & (labels >= num_classes)):
        raise ValueError(f"Valid target contains a label outside [0, {num_classes - 1}].")
    return torch.where(valid, labels, 0), valid


@torch.no_grad()
def prepare_blob_components(
    target: torch.Tensor,
    num_classes: int,
    *,
    instance_ids: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    ignore_label: int | None = -1,
    connectivity: int = 1,
    spatial_dims: int | None = None,
) -> BlobComponents:
    """Prepare CPU components for a 2D/3D batch, optionally using true instances.

    connectivity is the scipy convention: 1 means face neighbors (4/6), 2
    includes 2-axis diagonals (8/18), and 3 is full 26-connectivity in 3D.
    Positive instance IDs are namespaced by image AND semantic class, may be
    sparse/large, and must cover every valid foreground voxel. Missing IDs fail
    explicitly rather than silently discarding supervision. Background IDs are
    ignored. Set spatial_dims explicitly for a channel-less 3D target with depth
    one, which is otherwise ambiguous with [B, 1, H, W].
    """
    target = torch.as_tensor(target)
    if num_classes < 2:
        raise ValueError("num_classes must include background and at least one foreground class.")
    if spatial_dims is None:
        has_channel = target.ndim >= 4 and target.shape[1] == 1
        spatial_dims = target.ndim - (2 if has_channel else 1)
    if spatial_dims not in (2, 3):
        raise ValueError("Only 2D and 3D semantic batches are supported.")
    if connectivity not in range(1, spatial_dims + 1):
        raise ValueError(f"connectivity must be between 1 and {spatial_dims}.")
    shape = (target.shape[0], *target.shape[-spatial_dims:])
    semantic, valid = _semantic_and_valid(target, shape, num_classes, ignore_label, valid_mask)
    semantic_np = semantic.cpu().numpy()
    ids_np = None
    if instance_ids is not None:
        ids = _without_channel(instance_ids, shape, "instance_ids")
        foreground = (semantic > 0).to(ids.device)
        if ids.is_floating_point() and torch.any(foreground & (~torch.isfinite(ids) | (ids != ids.round()))):
            raise ValueError("Instance IDs must be finite integers on valid foreground voxels.")
        if torch.any(foreground & (ids <= 0)):
            raise ValueError("Positive instance IDs must cover all valid foreground voxels.")
        ids_np = ids.long().cpu().numpy()

    maps = np.zeros(shape, dtype=np.int64)
    all_classes, all_batches, all_sizes = [], [], []
    offset = 0
    structure = ndimage.generate_binary_structure(spatial_dims, connectivity)
    for b, labels in enumerate(semantic_np):
        if ids_np is None and cc3d is not None:
            adjacency = (4, 8)[connectivity - 1] if spatial_dims == 2 else (6, 18, 26)[connectivity - 1]
            local, count = cc3d.connected_components(
                np.ascontiguousarray(labels), connectivity=adjacency, return_N=True
            )
            sizes = np.bincount(local.ravel(), minlength=count + 1)[1:]
            # Each cc3d component has one semantic value. This obtains its class
            # in O(voxels + components) without a full image scan per component.
            class_sums = np.bincount(local.ravel(), weights=labels.ravel(), minlength=count + 1)[1:]
            classes = np.rint(class_sums / np.maximum(sizes, 1)).astype(np.int64)
        else:
            local = np.zeros(labels.shape, dtype=np.int64)
            class_list = []
            count = 0
            for c in np.unique(labels):
                if c == 0:
                    continue
                selected = labels == c
                if ids_np is None:
                    class_map, n = ndimage.label(selected, structure=structure)
                    local[selected] = class_map[selected] + count
                else:
                    unique, inverse = np.unique(ids_np[b][selected], return_inverse=True)
                    n = len(unique)
                    local[selected] = inverse + count + 1
                class_list.extend([int(c)] * n)
                count += n
            classes = np.asarray(class_list, dtype=np.int64)
            sizes = np.bincount(local.ravel(), minlength=count + 1)[1:]
        foreground = local > 0
        maps[b][foreground] = local[foreground].astype(np.int64) + offset
        all_classes.extend(classes.tolist())
        all_batches.extend([b] * count)
        all_sizes.extend(sizes.tolist())
        offset += count
    return BlobComponents(
        labels=torch.from_numpy(maps),
        class_ids=torch.tensor(all_classes, dtype=torch.long),
        batch_ids=torch.tensor(all_batches, dtype=torch.long),
        sizes=torch.tensor(all_sizes, dtype=torch.long),
        num_classes=int(num_classes),
    )


def _check_logits(logits: torch.Tensor) -> None:
    if logits.ndim not in (4, 5) or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("logits must be [B, C, H, W] or [B, C, D, H, W], B>0 and C>=2.")
    if not logits.is_floating_point():
        raise ValueError("logits must be floating point.")


def _zero(value):
    return value.reshape(-1)[:0].sum()


def _blob_from_probabilities(probs, semantic, valid, components, smooth):
    if components.num_classes != probs.shape[1] or tuple(components.labels.shape) != tuple(semantic.shape):
        raise ValueError("Prepared blob components do not match the logits/target shape or class count.")
    device = probs.device
    component_map = components.labels.to(device=device)
    classes = components.class_ids.to(device=device)
    batches = components.batch_ids.to(device=device)
    # Reject mismatched augmented targets/validity masks, including lost labels.
    lut = torch.cat((classes.new_zeros(1), classes))
    if not torch.equal(lut[component_map], semantic):
        raise ValueError("Prepared components must match this exact augmented semantic target and valid mask.")
    if classes.numel() == 0:
        return _zero(probs)

    correct_class = probs.gather(1, semantic[:, None]).squeeze(1)
    foreground = component_map > 0
    # Do not scatter the typically millions of background voxels into one
    # unused bin; on CUDA those unnecessary atomic additions would contend.
    totals = probs.new_zeros(classes.numel()).scatter_add(
        0, component_map[foreground] - 1, correct_class[foreground]
    )
    sizes = components.sizes.to(device=device, dtype=probs.dtype)
    image_losses = []
    # Loop only over images and semantic classes, never over individual blobs.
    # Compute background directly; subtracting two huge foreground sums would
    # erase small false-positive mass through FP32 cancellation.
    for b in components.batch_ids.unique().tolist():
        class_losses = []
        image_components = batches == b
        for c in components.class_ids[components.batch_ids == b].unique().tolist():
            selected = image_components & (classes == c)
            background = valid[b] & (semantic[b] != c)
            bg_mass = probs[b, c].masked_select(background).sum()
            tp = totals[selected]
            score = (2 * tp + smooth) / (tp + sizes[selected] + bg_mass + smooth).clamp_min(1e-8)
            class_losses.append((1 - score).mean())
        image_losses.append(torch.stack(class_losses).mean())
    return torch.stack(image_losses).mean()


class MulticlassBlobDiceLoss(nn.Module):
    """Dice-based blob term over softmax logits; absent classes/images are skipped."""

    def __init__(self, smooth=1e-5, ignore_label=-1, connectivity=1):
        super().__init__()
        if not math.isfinite(smooth) or smooth < 0:
            raise ValueError("smooth must be finite and nonnegative.")
        self.smooth, self.ignore_label, self.connectivity = float(smooth), ignore_label, connectivity

    def forward(self, logits, target, *, instance_ids=None, valid_mask=None, components=None):
        _check_logits(logits)
        if components is not None and instance_ids is not None:
            raise ValueError("Pass either prepared components or instance_ids, not both.")
        shape = (logits.shape[0], *logits.shape[2:])
        semantic, valid = _semantic_and_valid(target, shape, logits.shape[1], self.ignore_label, valid_mask)
        if components is None:
            components = prepare_blob_components(
                semantic, logits.shape[1], instance_ids=instance_ids, valid_mask=valid,
                ignore_label=None, connectivity=self.connectivity, spatial_dims=logits.ndim - 2,
            )
        semantic, valid = semantic.to(logits.device), valid.to(logits.device)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
            work = work.masked_fill(~valid[:, None], 0)
            return _blob_from_probabilities(work.softmax(1), semantic, valid, components, self.smooth)


class FlatCEAndDiceAndBlobLoss(nn.Module):
    """nnU-Net flat CE + negative foreground Dice + weighted blob Dice.

    For ordinary nonempty targets, weight_blob=0 matches DC_and_CE_loss with
    do_bg=False and the same smooth/batch_dice settings. All negative targets
    and an optional positive ignore_label are ignored. Entirely ignored images
    are excluded from per-image Dice rather than returning the constant -1.
    Half/bfloat16 logits are evaluated in FP32. Blob reduction is rank-local;
    only batch Dice uses cross-rank statistics when batch_dice AND ddp are true.
    Empty/absent foreground still receives the global CE/Dice penalty.
    """

    def __init__(self, weight_ce=1., weight_dice=1., weight_blob=1., smooth=1e-5,
                 ignore_label=-1, batch_dice=False, ddp=False, connectivity=1):
        super().__init__()
        if any(not math.isfinite(v) or v < 0 for v in (weight_ce, weight_dice, weight_blob, smooth)):
            raise ValueError("Loss weights and smooth must be finite and nonnegative.")
        self.weight_ce, self.weight_dice, self.weight_blob = float(weight_ce), float(weight_dice), float(weight_blob)
        self.smooth, self.ignore_label = float(smooth), ignore_label
        self.batch_dice, self.ddp, self.connectivity = bool(batch_dice), bool(ddp), connectivity

    def forward(self, logits, target, *, instance_ids=None, valid_mask=None, components=None):
        _check_logits(logits)
        if components is not None and instance_ids is not None:
            raise ValueError("Pass either prepared components or instance_ids, not both.")
        shape = (logits.shape[0], *logits.shape[2:])
        semantic, valid = _semantic_and_valid(target, shape, logits.shape[1], self.ignore_label, valid_mask)
        if self.weight_blob and components is None:
            components = prepare_blob_components(
                semantic, logits.shape[1], instance_ids=instance_ids, valid_mask=valid,
                ignore_label=None, connectivity=self.connectivity, spatial_dims=logits.ndim - 2,
            )
        semantic, valid = semantic.to(logits.device), valid.to(logits.device)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
            work = work.masked_fill(~valid[:, None], 0)
            result = _zero(work)
            if self.weight_ce:
                ce = F.cross_entropy(work, semantic, reduction="none")
                result = result + self.weight_ce * ce.masked_select(valid).sum() / valid.sum().clamp_min(1)
            if not (self.weight_dice or self.weight_blob):
                return result
            probs = work.softmax(1)  # Shared by the global and component terms.
            if self.weight_dice:
                flat_probs = probs.flatten(2)
                labels = semantic.flatten(1)
                mask = valid.flatten(1).to(probs.dtype)
                correct = flat_probs.gather(1, labels[:, None]).squeeze(1) * mask
                tp = probs.new_zeros(probs.shape[:2]).scatter_add(1, labels, correct)
                gt = probs.new_zeros(probs.shape[:2]).scatter_add(1, labels, mask)
                pred = torch.einsum("bcv,bv->bc", flat_probs, mask)
                if self.batch_dice:
                    if self.ddp:
                        # Keep collectives on every rank, including ranks with
                        # entirely ignored targets, to avoid DDP deadlocks.
                        tp = AllGatherGrad.apply(tp.sum(0)).sum(0)
                        pred = AllGatherGrad.apply(pred.sum(0)).sum(0)
                        gt = AllGatherGrad.apply(gt.sum(0)).sum(0)
                    else:
                        tp, pred, gt = tp.sum(0), pred.sum(0), gt.sum(0)
                    if gt.sum() > 0:
                        dice = (2 * tp[1:] + self.smooth) / (pred[1:] + gt[1:] + self.smooth).clamp_min(1e-8)
                        result = result - self.weight_dice * dice.mean()
                else:
                    nonempty = valid.flatten(1).any(1)
                    if nonempty.any():
                        dice = (2 * tp[nonempty, 1:] + self.smooth) / (
                            pred[nonempty, 1:] + gt[nonempty, 1:] + self.smooth
                        ).clamp_min(1e-8)
                        result = result - self.weight_dice * dice.mean()
            if self.weight_blob:
                result = result + self.weight_blob * _blob_from_probabilities(
                    probs, semantic, valid, components, self.smooth
                )
            return result
