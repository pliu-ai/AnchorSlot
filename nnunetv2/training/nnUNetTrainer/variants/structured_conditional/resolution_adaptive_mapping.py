from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .hierarchical_parallel_mapping import NUM_ORIGINAL_CLASSES


ATOMIC_NAMES: Tuple[str, ...] = (
    "ecs", "pm", "cyto", "mito_mem", "mito_lum", "mito_ribo",
    "golgi_mem", "golgi_lum", "ves_mem", "ves_lum", "endo_mem",
    "endo_lum", "lyso_mem", "lyso_lum", "ld_mem", "ld_lum",
    "er_mem", "er_lum", "eres_mem", "eres_lum", "ne_mem", "ne_lum",
    "np_out", "np_in", "hchrom", "echrom", "nucpl", "mt_out",
    "mt_in", "perox_mem", "perox_lum",
)
ATOMIC_NAME_TO_LABEL = {name: index + 1 for index, name in enumerate(ATOMIC_NAMES)}

PARENT_CHILDREN = {
    "mito": ("mito_mem", "mito_lum", "mito_ribo"),
    "golgi": ("golgi_mem", "golgi_lum"),
    "ves": ("ves_mem", "ves_lum"),
    "endo": ("endo_mem", "endo_lum"),
    "lyso": ("lyso_mem", "lyso_lum"),
    "ld": ("ld_mem", "ld_lum"),
    "perox": ("perox_mem", "perox_lum"),
    "eres": ("eres_mem", "eres_lum"),
    "mt": ("mt_in", "mt_out"),
    "np": ("np_in", "np_out"),
    "chrom": ("hchrom", "echrom"),
    "ne": ("ne_mem", "ne_lum", "np_in", "np_out"),
    "ne_mem_all": ("ne_mem", "np_in", "np_out"),
    "nuc": ("nucpl", "hchrom", "echrom", "ne_mem", "ne_lum", "np_in", "np_out"),
    "er_mem_all": ("er_mem", "ne_mem", "eres_mem"),
    "er": ("er_mem", "er_lum", "ne_mem", "ne_lum", "np_in", "np_out", "eres_mem", "eres_lum"),
    "cell": tuple(name for name in ATOMIC_NAMES if name != "ecs"),
}
PARENT_NAMES: Tuple[str, ...] = tuple(PARENT_CHILDREN)
NUM_PARENT_CLASSES = len(PARENT_NAMES)
INSTANCE_PARENT_NAMES: Tuple[str, ...] = (
    "nuc", "ves", "endo", "lyso", "ld", "perox", "mito", "np", "mt", "cell"
)
PARENT_LABEL_IDS: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(ATOMIC_NAME_TO_LABEL[name] for name in PARENT_CHILDREN[parent])
    for parent in PARENT_NAMES
)

# A mutually exclusive family code used only for the weak separation auxiliary
# target when native instance IDs are unavailable. Cell is deliberately omitted
# because it overlaps all of the more specific families.
_SEPARATION_FAMILIES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("mito", PARENT_CHILDREN["mito"]),
    ("ves", PARENT_CHILDREN["ves"]),
    ("endo", PARENT_CHILDREN["endo"]),
    ("lyso", PARENT_CHILDREN["lyso"]),
    ("ld", PARENT_CHILDREN["ld"]),
    ("perox", PARENT_CHILDREN["perox"]),
    ("np", PARENT_CHILDREN["np"]),
    ("mt", PARENT_CHILDREN["mt"]),
    ("nuc", ("ne_mem", "ne_lum", "hchrom", "echrom", "nucpl")),
)
LABEL_TO_FAMILY = torch.zeros(NUM_ORIGINAL_CLASSES, dtype=torch.long)
for family_id, (_, children) in enumerate(_SEPARATION_FAMILIES, start=1):
    for child in children:
        LABEL_TO_FAMILY[ATOMIC_NAME_TO_LABEL[child]] = family_id


def build_parent_targets(
    segmentation: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert exclusive atomic IDs to the 17 overlapping CellMap parent masks."""
    labels = segmentation[:, 0].long() if segmentation.shape[1] == 1 else segmentation.long()
    targets = []
    for label_ids in PARENT_LABEL_IDS:
        member = torch.zeros_like(labels, dtype=torch.bool)
        for label_id in label_ids:
            member |= labels == label_id
        targets.append(member)
    parent = torch.stack(targets, dim=1).float()
    if valid_mask is not None:
        parent = parent * valid_mask.float()
    return parent


def derived_parent_probabilities(semantic_logits: torch.Tensor) -> torch.Tensor:
    """Marginalize the exclusive 32-way leaf distribution into DAG parents."""
    probabilities = torch.softmax(semantic_logits, dim=1)
    return torch.stack(
        [probabilities[:, list(label_ids)].sum(dim=1) for label_ids in PARENT_LABEL_IDS],
        dim=1,
    )


def active_parent_mask(active_semantic: torch.Tensor) -> torch.Tensor:
    """A parent is supervised only when its native parent annotation is active."""
    if active_semantic.ndim != 2 or active_semantic.shape[1] != NUM_ORIGINAL_CLASSES:
        raise ValueError("active_semantic must have shape [B, 32]")
    return torch.stack(
        [active_semantic[:, list(label_ids)].any(dim=1) for label_ids in PARENT_LABEL_IDS],
        dim=1,
    )


def build_instance_separation_targets(
    segmentation: torch.Tensor,
    valid_mask: torch.Tensor,
    instance_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build one boundary plus one nearest-neighbor affinity target per axis.

    Native instance IDs take priority. With merged atomic labels, a conservative
    family target is used as an auxiliary edge prior; it must not be described as
    true instance supervision in an ablation.
    """
    labels = segmentation[:, 0].long()
    valid = valid_mask[:, 0].bool()
    if instance_ids is not None:
        ids = instance_ids[:, 0].long()
        # Combine organelle family and native ID so equal numeric IDs from two
        # classes never become a positive affinity.
        family = LABEL_TO_FAMILY.to(labels.device)[labels.clamp(0, NUM_ORIGINAL_CLASSES - 1)]
        codes = family * (ids.max().clamp_min(1) + 1) + ids
        foreground = ids > 0
    else:
        codes = LABEL_TO_FAMILY.to(labels.device)[labels.clamp(0, NUM_ORIGINAL_CLASSES - 1)]
        foreground = codes > 0

    spatial_dims = labels.ndim - 1
    boundary = torch.zeros_like(labels, dtype=torch.bool)
    boundary_valid = torch.zeros_like(labels, dtype=torch.bool)
    affinities = []
    affinity_valid = []
    for spatial_axis in range(spatial_dims):
        axis = spatial_axis + 1
        left = [slice(None)] * labels.ndim
        right = [slice(None)] * labels.ndim
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        left_t, right_t = tuple(left), tuple(right)
        pair_valid = valid[left_t] & valid[right_t]
        pair_foreground = foreground[left_t] | foreground[right_t]
        known = pair_valid & pair_foreground
        same = (codes[left_t] == codes[right_t]) & foreground[left_t] & foreground[right_t]
        difference = known & ~same
        boundary[left_t] |= difference
        boundary[right_t] |= difference
        boundary_valid[left_t] |= known
        boundary_valid[right_t] |= known

        affinity = torch.zeros_like(labels, dtype=torch.float32)
        affinity_mask = torch.zeros_like(labels, dtype=torch.bool)
        affinity[left_t] = same.float()
        affinity_mask[left_t] = known
        affinities.append(affinity)
        affinity_valid.append(affinity_mask)

    target = torch.cat(
        [boundary[:, None].float(), torch.stack(affinities, dim=1)], dim=1
    )
    mask = torch.cat(
        [boundary_valid[:, None], torch.stack(affinity_valid, dim=1)], dim=1
    )
    return target, mask


def resize_binary_targets(
    target: torch.Tensor,
    mask: torch.Tensor,
    shape: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if tuple(target.shape[2:]) == tuple(shape):
        return target, mask
    target = F.interpolate(target, size=tuple(shape), mode="nearest")
    mask = F.interpolate(mask.float(), size=tuple(shape), mode="nearest") > 0.5
    return target, mask
