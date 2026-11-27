"""SALT's tree softmax and reachability-encoded CE + Dice for CellMap.

Sources: arXiv:2407.08878, Sec. 2.2--2.3; UMEssen/SALT commit
3e36a085a7167afd5a7660df8b8f5b5d059e9854, salt/core/activations.py and
salt/train.py:_compute_loss. This is an independent, numerically stable torch
implementation. It uses positive logits as in the official code (the paper's
displayed softmax has a minus sign). The original medical label tree, sink
labels, MONAI backbone and CT preprocessing are replaced by the CellMap tree
and shared nnU-Net experiment protocol. Voxels explicitly masked/ignored are
excluded; an unannotated class cannot be inferred from a background label.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .hierarchy import (LabelHierarchy, SALT_HIERARCHY, SALT_NUM_LOGITS,
                        graph_zero, prepare_target)


def _salt_conditional_log_probs(logits: torch.Tensor, hierarchy: LabelHierarchy) -> Tuple[torch.Tensor, ...]:
    """Sibling-normalized node views sharing their log-softmax output storage."""
    if logits.ndim < 3 or logits.shape[1] != hierarchy.num_nodes:
        raise ValueError(f"SALT expects [B,{hierarchy.num_nodes},...] logits")
    work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    conditional = [None] * hierarchy.num_nodes
    for _, children in hierarchy.children:
        values = F.log_softmax(work[:, children], dim=1).unbind(1)
        for node, value in zip(children, values):
            conditional[node] = value
    return tuple(conditional)


def salt_node_log_probs(logits: torch.Tensor, hierarchy: LabelHierarchy = SALT_HIERARCHY) -> torch.Tensor:
    """Sibling log-softmax followed by root-to-node log-probability sums."""
    conditional = _salt_conditional_log_probs(logits, hierarchy)
    result = []
    for path in hierarchy.paths:
        value = conditional[path[0]]
        for node in path[1:]:
            value = value + conditional[node]
        result.append(value)
    return torch.stack(result, dim=1)


def salt_leaf_log_probs(logits: torch.Tensor, hierarchy: LabelHierarchy = SALT_HIERARCHY) -> torch.Tensor:
    """Return original-ID leaf log probabilities, normalized over the leaves."""
    return salt_node_log_probs(logits, hierarchy)[:, :hierarchy.num_leaves]


def _salt_node_terms(labels: torch.Tensor, leaves: Tuple[int, ...], smooth: float,
                     *path_conditionals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return scalar CE and Dice fraction; dense temporaries can be recomputed."""
    log_probability = path_conditionals[0]
    for conditional in path_conditionals[1:]:
        log_probability = log_probability + conditional
    truth = torch.zeros_like(labels, dtype=torch.bool)
    for leaf in leaves:
        truth = truth | (labels == leaf)
    ce = -log_probability[truth].sum() / labels.numel()
    probability = log_probability.exp()
    intersection = probability[truth].sum()
    dice_fraction = 2 * intersection / (probability.sum() + truth.sum() + smooth)
    return ce, dice_fraction


class SALTLoss(nn.Module):
    """The published hybrid objective over all non-root hierarchy nodes.

    CE is the sum of positive ancestor-node negative log probabilities, not
    merely leaf NLL. Dice includes background, internal nodes and absent nodes,
    with zero numerator smoothing, matching the official training code. Losses
    are averaged per nonempty sample, as in its per-case loss computation.
    Non-reentrant per-node checkpointing recomputes path sums and dense Dice
    intermediates from shared sibling log-softmax outputs. It changes memory/
    runtime, not the objective or parameters. There is no second persistent
    dense stack of marginal log probabilities in the loss path.
    """
    def __init__(self, ignore_label: Optional[int] = -1, ce_weight: float = 1.0,
                 dice_weight: float = 1.0, smooth: float = 1e-5,
                 hierarchy: LabelHierarchy = SALT_HIERARCHY,
                 checkpoint_nodes: bool = True):
        super().__init__()
        if ce_weight < 0 or dice_weight < 0 or smooth <= 0:
            raise ValueError("loss weights must be nonnegative and smooth positive")
        self.ignore_label = ignore_label
        self.ce_weight, self.dice_weight, self.smooth = ce_weight, dice_weight, smooth
        self.hierarchy = hierarchy
        self.checkpoint_nodes = bool(checkpoint_nodes)

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        target, valid = prepare_target(logits, target, valid_mask, self.ignore_label, self.hierarchy.num_leaves)
        if logits.shape[1] != self.hierarchy.num_nodes:
            raise ValueError(f"SALT requires {self.hierarchy.num_nodes} logits")
        per_sample = []
        for batch in range(logits.shape[0]):
            keep = valid[batch].reshape(-1)
            if not bool(keep.any()):
                continue
            # Filter before activation; ignored values cannot affect the loss.
            selected = logits[batch].flatten(1)[:, keep].unsqueeze(0)
            conditional = tuple(value[0] for value in _salt_conditional_log_probs(selected, self.hierarchy))
            labels = target[batch].reshape(-1)[keep]
            ce = graph_zero(selected)
            dice = graph_zero(selected)
            recompute = self.checkpoint_nodes and torch.is_grad_enabled() and selected.requires_grad
            for node, leaves in enumerate(self.hierarchy.leaf_descendants):
                # Pass immutable node metadata explicitly: backward recomputation
                # must never close over the changing loop variable.
                args = (labels, leaves, self.smooth,
                        *(conditional[ancestor] for ancestor in self.hierarchy.paths[node]))
                if recompute:
                    node_ce, fraction = checkpoint(_salt_node_terms, *args, use_reentrant=False,
                                                    preserve_rng_state=False)
                else:
                    node_ce, fraction = _salt_node_terms(*args)
                ce = ce + node_ce
                dice = dice + 1 - fraction
            per_sample.append(self.ce_weight * ce + self.dice_weight * dice / self.hierarchy.num_nodes)
        return torch.stack(per_sample).mean() if per_sample else graph_zero(logits)
