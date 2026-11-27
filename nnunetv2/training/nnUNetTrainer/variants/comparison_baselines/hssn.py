"""Paper-form HSSN: focal tree-min segmentation + tree-triplet learning.

Li et al., CVPR 2022, arXiv:2203.14335, Eqs. (3), (6)--(8), Sec. 3.3.
Consulted official HieraSeg cc3c1cfbabe3cc2af620e0193a245822cca8a841.
The released Cityscapes loss differs from the equations (additional CE, fixed
margin, differently scaled cosine distance); this implementation explicitly
follows the paper equations. CellMap uses 32 leaf and 18 coarse nodes, with
singleton coarse parents for background/anchors to maintain equal path depth.
The experiment adapter replaces the original backbone and 2D preprocessing.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .hierarchy import (HSSN_HIERARCHY, HSSN_NUM_LOGITS, LabelHierarchy,
                        graph_zero, prepare_target)


def hssn_leaf_scores(logits: torch.Tensor, hierarchy: LabelHierarchy = HSSN_HIERARCHY) -> torch.Tensor:
    """Eq. (3): sum sigmoid scores along each path, in original label order.

    These are decision scores, NOT normalized leaf probabilities. The implicit
    root contributes the same constant to all paths and can be omitted. A
    downstream softmax preserves argmax but does not calibrate these scores.
    """
    if logits.ndim < 3 or logits.shape[1] != hierarchy.num_nodes:
        raise ValueError(f"HSSN expects [B,{hierarchy.num_nodes},...] logits")
    work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    probabilities = work.sigmoid()
    return torch.stack([probabilities[:, path].sum(1)
                        for path in hierarchy.paths[:hierarchy.num_leaves]], dim=1)


class HSSNProjectionHead(nn.Module):
    """Two pointwise convolutions, ReLU, 256-d normalized pixel embeddings.

    GroupNorm replaces the official code's BatchNorm for small 3D batches; this
    is an explicit backbone-adaptation choice, not part of Eqs. (6)--(8).
    Set norm_groups=0 for the two-convolution/no-normalization form described
    in the paper. Run on a coarse feature grid (the adapter uses 1/4 scale).
    """
    def __init__(self, in_channels: int, projection_dim: int = 256,
                 spatial_dims: int = 3, norm_groups: int = 1):
        super().__init__()
        if spatial_dims not in (1, 2, 3) or in_channels < 1 or projection_dim < 1:
            raise ValueError("invalid projection dimensions")
        if norm_groups < 0 or (norm_groups and in_channels % norm_groups):
            raise ValueError("norm_groups must divide in_channels, or be zero")
        convolution = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[spatial_dims - 1]
        layers = [convolution(in_channels, in_channels, 1)]
        if norm_groups:
            layers.append(nn.GroupNorm(norm_groups, in_channels))
        layers.extend((nn.ReLU(inplace=False), convolution(in_channels, projection_dim, 1)))
        self.projection = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(features).float(), dim=1, eps=1e-12)


def hssn_triplet_weight(progress: float, maximum: float = 0.5) -> float:
    """Cosine warm-up 0 -> maximum, matching the released schedule direction."""
    if not math.isfinite(progress) or not 0 <= progress <= 1 or maximum < 0:
        raise ValueError("progress must be in [0, 1] and maximum nonnegative")
    return maximum * 0.5 * (1 - math.cos(math.pi * progress))


def _triplet_values(embeddings: torch.Tensor, labels: torch.Tensor, triplets: torch.Tensor,
                    hierarchy: LabelHierarchy, margin_epsilon: float) -> torch.Tensor:
    if embeddings.ndim != 2 or labels.ndim != 1 or embeddings.shape[0] != labels.shape[0]:
        raise ValueError("expected embeddings [N,C] and labels [N]")
    if triplets.ndim != 2 or triplets.shape[1] != 3:
        raise ValueError("triplets must have shape [T,3]")
    if triplets.numel() == 0:
        return embeddings.reshape(-1)[:0].float()
    if bool(((triplets < 0) | (triplets >= labels.numel())).any()):
        raise ValueError("triplet index outside embeddings")
    triplet_labels = labels[triplets]
    if bool(((triplet_labels < 0) | (triplet_labels >= hierarchy.num_leaves)).any()):
        raise ValueError("triplets may not include ignored/invalid labels")
    distances = hierarchy.leaf_distances(embeddings.device)
    a_label, p_label, n_label = triplet_labels.unbind(1)
    positive_distance = distances[a_label, p_label]
    negative_distance = distances[a_label, n_label]
    if bool((positive_distance >= negative_distance).any()):
        raise ValueError("triplets must satisfy tree distance(anchor,positive) < distance(anchor,negative)")
    if bool(((triplets[:, 0] == triplets[:, 1]) | (triplets[:, 0] == triplets[:, 2])).any()):
        raise ValueError("triplets must use distinct anchor and positive/negative pixels")
    vectors = F.normalize(embeddings.float(), dim=1, eps=1e-12)
    anchor, positive, negative = (vectors[triplets[:, i]] for i in range(3))
    d_positive = 0.5 * (1 - (anchor * positive).sum(1))
    d_negative = 0.5 * (1 - (anchor * negative).sum(1))
    margin = margin_epsilon + 0.5 * (negative_distance - positive_distance).float() / (2 * hierarchy.height)
    return F.relu(d_positive - d_negative + margin)


def hssn_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, triplets: torch.Tensor,
                      hierarchy: LabelHierarchy = HSSN_HIERARCHY,
                      margin_epsilon: float = 0.1) -> torch.Tensor:
    """Eqs. (7)--(8) for explicit pixel triplets, useful for reproducible checks."""
    values = _triplet_values(embeddings, labels, triplets, hierarchy, margin_epsilon)
    return values.mean() if values.numel() else graph_zero(embeddings)


def sample_hssn_triplets(labels: torch.Tensor, hierarchy: LabelHierarchy = HSSN_HIERARCHY,
                         max_triplets_per_class: int = 200) -> torch.Tensor:
    """Bounded class-balanced sampling from a flattened batch of voxel labels.

    Samples all valid tree-distance orderings, including distinct same-class
    positives (allowed by the paper, omitted by the released implementation).
    Sampling is with replacement within each class and uses torch's RNG. The
    number of anchors is capped at 200 per class, as in the released code.
    Ignored labels must be -1. Mining is local to the current distributed rank.
    """
    if labels.ndim != 1 or max_triplets_per_class < 1:
        raise ValueError("expected flat labels and a positive triplet cap")
    if bool(((labels < -1) | (labels >= hierarchy.num_leaves)).any()):
        raise ValueError("sampling labels must be valid leaf IDs or -1")
    classes = torch.unique(labels[labels >= 0])
    if classes.numel() < 2:
        return torch.empty((0, 3), device=labels.device, dtype=torch.long)
    pools, alternatives, sizes = [], [], []
    for label in classes:
        indices = torch.where(labels == label)[0]
        draws = torch.randint(indices.numel(), (max_triplets_per_class,), device=labels.device)
        pools.append(indices[draws])
        # The alternate is distinct whenever a same-class positive is possible.
        offset = torch.randint(1, max(2, indices.numel()), draws.shape, device=labels.device)
        alternatives.append(indices[(draws + offset) % indices.numel()])
        sizes.append(indices.numel())
    pools = torch.stack(pools)
    alternatives = torch.stack(alternatives)
    distances = hierarchy.leaf_distances(labels.device)[classes[:, None], classes[None, :]]
    triplets = []
    for anchor_index, count in enumerate(sizes):
        possible = distances[anchor_index, :, None] < distances[anchor_index, None, :]
        if count < 2:
            possible[anchor_index] = False
        pairs = torch.nonzero(possible, as_tuple=False)
        if not pairs.numel():
            continue
        number = min(count, max_triplets_per_class)
        chosen = pairs[torch.randint(len(pairs), (number,), device=labels.device)]
        columns = torch.arange(number, device=labels.device)
        positive = pools[chosen[:, 0], columns]
        positive = torch.where(chosen[:, 0] == anchor_index,
                               alternatives[anchor_index, :number], positive)
        triplets.append(torch.stack((pools[anchor_index, :number], positive,
                                     pools[chosen[:, 1], columns]), dim=1))
    return torch.cat(triplets) if triplets else torch.empty((0, 3), device=labels.device, dtype=torch.long)


def _feature_targets(features: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if features.shape[0] != labels.shape[0] or features.ndim != labels.ndim + 1:
        raise ValueError("feature batch/spatial dimensionality must match target")
    labels = labels.masked_fill(~valid, -1)
    if features.shape[2:] != labels.shape[1:]:
        labels = F.interpolate(labels[:, None].float(), size=features.shape[2:], mode="nearest")[:, 0].long()
    return labels.reshape(-1)


def _hssn_node_loss(work: torch.Tensor, leaf: torch.Tensor, ancestors: Tuple[int, ...],
                    descendants: Tuple[int, ...], positive_leaves: Tuple[int, ...],
                    gamma: float, number_valid: torch.Tensor) -> torch.Tensor:
    """Eq. (6) for one node, returning only its reduced scalar contribution."""
    # Sigmoid is monotonic, so min/max logits gives exactly Eq. (4) while
    # softplus evaluates log probabilities without clipping. work is FP32.
    positive = work[list(ancestors)].amin(0)
    negative = work[list(descendants)].amax(0)
    truth = torch.zeros_like(leaf, dtype=torch.bool)
    for child in positive_leaves:
        truth = truth | (leaf == child)
    positive_loss = positive.neg().sigmoid().pow(gamma) * F.softplus(-positive)
    negative_loss = negative.sigmoid().pow(gamma) * F.softplus(negative)
    return torch.where(truth, positive_loss, negative_loss).sum() / number_valid


class HSSNLoss(nn.Module):
    """Focal tree-min (Eq. 6) plus projected-feature tree-triplet (Eqs. 7--8).

    Segmentation sums node losses and averages valid voxels, directly following
    Eq. (6). It does not add the released Cityscapes code's extra CE/scale factor.
    For deep supervision call segmentation_loss on auxiliary scales, and apply
    the triplet term once to the main prediction's projected features.
    Non-reentrant per-node checkpointing avoids saving each node's dense focal
    intermediates. The shared FP32 logits and the mathematical loss are unchanged.
    """
    def __init__(self, ignore_label: Optional[int] = -1, gamma: float = 2.0,
                 triplet_weight: float = 0.5, max_triplets_per_class: int = 200,
                 margin_epsilon: float = 0.1, hierarchy: LabelHierarchy = HSSN_HIERARCHY,
                 checkpoint_nodes: bool = True):
        super().__init__()
        if gamma < 0 or triplet_weight < 0 or margin_epsilon < 0 or max_triplets_per_class < 1:
            raise ValueError("invalid HSSN loss parameters")
        self.ignore_label, self.gamma, self.triplet_weight = ignore_label, gamma, triplet_weight
        self.max_triplets_per_class, self.margin_epsilon = max_triplets_per_class, margin_epsilon
        self.hierarchy = hierarchy
        self.checkpoint_nodes = bool(checkpoint_nodes)

    def segmentation_loss(self, logits: torch.Tensor, target: torch.Tensor,
                          valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if logits.ndim < 3 or logits.shape[1] != self.hierarchy.num_nodes:
            raise ValueError(f"HSSN requires {self.hierarchy.num_nodes} node logits")
        labels, valid = prepare_target(logits, target, valid_mask, self.ignore_label, self.hierarchy.num_leaves)
        number_valid = valid.sum()
        if not bool(number_valid):
            return graph_zero(logits)
        loss = graph_zero(logits)
        for batch in range(logits.shape[0]):
            keep = valid[batch].reshape(-1)
            if not bool(keep.any()):
                continue
            work = logits[batch].flatten(1)[:, keep].float()
            leaf = labels[batch].reshape(-1)[keep]
            recompute = self.checkpoint_nodes and torch.is_grad_enabled() and work.requires_grad
            for node, ancestors in enumerate(self.hierarchy.paths):
                # Explicit tuple arguments avoid late-bound loop variables when
                # checkpoint invokes this function again during backward.
                args = (work, leaf, ancestors, self.hierarchy.descendants[node],
                        self.hierarchy.leaf_descendants[node], self.gamma, number_valid)
                if recompute:
                    node_loss = checkpoint(_hssn_node_loss, *args, use_reentrant=False,
                                           preserve_rng_state=False)
                else:
                    node_loss = _hssn_node_loss(*args)
                loss = loss + node_loss
        return loss

    def representation_loss(self, features: torch.Tensor, target: torch.Tensor,
                            valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Shape checking occurs at target resolution, independently of the
        # projected feature grid. Empty placeholder carries no feature values.
        target_shape = target.shape[2:] if target.ndim == features.ndim else target.shape[1:]
        reference = features.new_empty((features.shape[0], 1, *target_shape))
        labels, valid = prepare_target(reference, target, valid_mask, self.ignore_label, self.hierarchy.num_leaves)
        labels = _feature_targets(features, labels, valid)
        triplets = sample_hssn_triplets(labels, self.hierarchy, self.max_triplets_per_class)
        if not triplets.numel():
            return graph_zero(features)
        # Gather only selected pixels; never create a full contiguous N x C copy.
        indices = triplets.reshape(-1)
        spatial_size = math.prod(features.shape[2:])
        sampled = features.flatten(2)[indices // spatial_size, :, indices % spatial_size]
        sampled_labels = labels[indices]
        local_triplets = torch.arange(indices.numel(), device=features.device).reshape(-1, 3)
        values = _triplet_values(sampled, sampled_labels, local_triplets, self.hierarchy, self.margin_epsilon)
        # Equal per-anchor-class weight, as in the released bounded sampler.
        anchor_classes = labels[triplets[:, 0]]
        sums = values.new_zeros(self.hierarchy.num_leaves).scatter_add(0, anchor_classes, values)
        counts = torch.bincount(anchor_classes, minlength=self.hierarchy.num_leaves)
        return (sums[counts > 0] / counts[counts > 0]).mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor, features: Optional[torch.Tensor] = None,
                valid_mask: Optional[torch.Tensor] = None, progress: float = 1.0) -> torch.Tensor:
        loss = self.segmentation_loss(logits, target, valid_mask)
        if features is None:
            # The trainer composes DS segmentation terms and calls feature_loss
            # exactly once. FTM alone must not be reported as the full method.
            return loss
        return loss + self.feature_loss(features, target, valid_mask=valid_mask, progress=progress)

    def feature_loss(self, features: torch.Tensor, target: torch.Tensor,
                     valid_mask: Optional[torch.Tensor] = None, progress: float = 1.0) -> torch.Tensor:
        """Weighted TT term, called once outside the DS segmentation sum."""
        weight = hssn_triplet_weight(progress, self.triplet_weight)
        # Evaluate even at weight=0, retaining DDP graph participation for the
        # projection head at the start of its cosine warm-up.
        return weight * self.representation_loss(features, target, valid_mask)
