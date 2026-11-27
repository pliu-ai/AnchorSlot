"""Fixed CellMap trees for the SALT and HSSN comparison adaptations.

These are label-group trees, not claims about biological containment. Leaves
0..31 retain the dataset IDs. Root is implicit, with index -1. The 13 grouping
nodes follow the fixed ER-dynamic grouping, independent of environment overrides.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Tuple

import torch

from .taxonomy import DYNAMIC_GROUP_LABELS, DYNAMIC_GROUP_NAMES, LEAF_NAMES


@dataclass(frozen=True)
class LabelHierarchy:
    parents: Tuple[int, ...]
    num_leaves: int
    names: Tuple[str, ...] = ()

    def __post_init__(self):
        if not 0 < self.num_leaves <= len(self.parents):
            raise ValueError("num_leaves must be in 1..num_nodes")
        if self.names and len(self.names) != len(self.parents):
            raise ValueError("names must have one entry per non-root node")
        for node, parent in enumerate(self.parents):
            if parent < -1 or parent >= len(self.parents) or parent == node:
                raise ValueError("invalid hierarchy parent")
            seen = {node}
            while parent != -1:
                if parent in seen:
                    raise ValueError("hierarchy contains a cycle")
                seen.add(parent)
                parent = self.parents[parent]
        actual_leaves = {i for i in range(len(self.parents)) if i not in self.parents}
        if actual_leaves != set(range(self.num_leaves)):
            raise ValueError("leaves must occupy exactly channels 0..num_leaves-1")

    @property
    def num_nodes(self) -> int:
        return len(self.parents)

    @cached_property
    def paths(self) -> Tuple[Tuple[int, ...], ...]:
        paths = []
        for node in range(self.num_nodes):
            path = []
            while node != -1:
                path.append(node)
                node = self.parents[node]
            paths.append(tuple(reversed(path)))
        return tuple(paths)

    @cached_property
    def children(self) -> Tuple[Tuple[int, Tuple[int, ...]], ...]:
        return tuple((parent, tuple(i for i, p in enumerate(self.parents) if p == parent))
                     for parent in sorted(set(self.parents)))

    @cached_property
    def descendants(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(tuple(j for j, path in enumerate(self.paths) if i in path)
                     for i in range(self.num_nodes))

    @cached_property
    def leaf_descendants(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(tuple(j for j in descendants if j < self.num_leaves)
                     for descendants in self.descendants)

    @property
    def height(self) -> int:
        return max(map(len, self.paths))

    def leaf_distances(self, device=None) -> torch.Tensor:
        # Include the implicit root when paths do not share an explicit node.
        distances = []
        for path_a in self.paths[:self.num_leaves]:
            row = []
            for path_b in self.paths[:self.num_leaves]:
                common = 0
                for a, b in zip(path_a, path_b):
                    if a != b:
                        break
                    common += 1
                row.append(len(path_a) + len(path_b) - 2 * common)
            distances.append(row)
        return torch.tensor(distances, device=device, dtype=torch.long)


def cellmap_hierarchy(*, balanced: bool = False) -> LabelHierarchy:
    """SALT: 45 nodes; HSSN: 50 with five singleton coarse categories.

    HSSN uses equal-depth paths, as in its published benchmark hierarchies, so
    summing sigmoid scores does not systematically favor deeper leaf classes.
    No extra semantic labels are introduced: singleton nodes duplicate the
    background/anchor concepts at the coarse level.
    """
    parents = [-1] * 45
    names = list(LEAF_NAMES) + list(DYNAMIC_GROUP_NAMES)
    for group_id, labels in enumerate(DYNAMIC_GROUP_LABELS):
        for label in labels:
            parents[label] = 32 + group_id
    fixed = ((0, "background"), (1, "ecs"), (2, "pm"), (3, "cyto"), (27, "nucpl"))
    for label, name in fixed:
        names[label] = name
        if balanced:
            parents[label] = len(parents)
            parents.append(-1)
            names.append(name + "_coarse")
    return LabelHierarchy(tuple(parents), 32, tuple(names))


SALT_HIERARCHY = cellmap_hierarchy()
HSSN_HIERARCHY = cellmap_hierarchy(balanced=True)
SALT_NUM_LOGITS = SALT_HIERARCHY.num_nodes
HSSN_NUM_LOGITS = HSSN_HIERARCHY.num_nodes


def prepare_target(
    reference: torch.Tensor, target: torch.Tensor, valid_mask: Optional[torch.Tensor],
    ignore_label: Optional[int], num_leaves: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize leaf labels and voxel validity without modifying caller data."""
    if target.ndim == reference.ndim:
        if target.shape[1] != 1:
            raise ValueError("target channel dimension must be one")
        target = target[:, 0]
    expected = (reference.shape[0], *reference.shape[2:])
    if tuple(target.shape) != expected:
        raise ValueError(f"target shape {tuple(target.shape)} does not match {expected}")
    target = target.to(device=reference.device, dtype=torch.long)
    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore_label is not None:
        valid = valid & (target != ignore_label)
    if valid_mask is not None:
        if valid_mask.ndim == reference.ndim:
            if valid_mask.shape[1] != 1:
                raise ValueError("valid_mask channel dimension must be one")
            valid_mask = valid_mask[:, 0]
        if tuple(valid_mask.shape) != expected:
            raise ValueError("valid_mask must match the target shape")
        valid = valid & valid_mask.to(device=reference.device, dtype=torch.bool)
    if bool((valid & ((target < 0) | (target >= num_leaves))).any()):
        raise ValueError("valid target contains a label outside the leaf label set")
    return target.masked_fill(~valid, 0), valid


def graph_zero(tensor: torch.Tensor) -> torch.Tensor:
    """A differentiable zero even when ignored entries contain Inf/NaN."""
    return tensor.reshape(-1)[:0].sum().float()
