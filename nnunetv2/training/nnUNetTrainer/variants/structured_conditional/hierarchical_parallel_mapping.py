from __future__ import annotations

from typing import Optional, Tuple

import torch

from .label_mapping_no_slot3 import DYNAMIC_GROUP_SPECS, FIXED_ORIGINAL_TO_OUTPUT


NUM_ORIGINAL_CLASSES = 32
NUM_PARALLEL_GROUPS = len(DYNAMIC_GROUP_SPECS)
NUM_ANCHOR_SLOTS = 2

# Keep fixed classes as leaves in the coarse taxonomy. Dynamic subclasses first
# select their organelle/group and are then resolved by a reusable anchor slot.
FIXED_ORIGINAL_LABELS: Tuple[int, ...] = tuple(sorted(FIXED_ORIGINAL_TO_OUTPUT))
BACKGROUND_COARSE_CHANNEL = 0
FIXED_COARSE_START = 1
GROUP_COARSE_START = FIXED_COARSE_START + len(FIXED_ORIGINAL_LABELS)
NUM_COARSE_CHANNELS = GROUP_COARSE_START + NUM_PARALLEL_GROUPS

ORIGINAL_TO_COARSE = [BACKGROUND_COARSE_CHANNEL] * NUM_ORIGINAL_CLASSES
ORIGINAL_TO_GROUP = [-1] * NUM_ORIGINAL_CLASSES
ORIGINAL_TO_SLOT = [-1] * NUM_ORIGINAL_CLASSES

for coarse_channel, original_label in enumerate(FIXED_ORIGINAL_LABELS, start=FIXED_COARSE_START):
    ORIGINAL_TO_COARSE[original_label] = coarse_channel

for spec in DYNAMIC_GROUP_SPECS:
    if spec.num_slots > NUM_ANCHOR_SLOTS:
        raise ValueError(
            f"group {spec.display_name!r} has {spec.num_slots} subclasses, "
            f"but only {NUM_ANCHOR_SLOTS} anchor slots are configured"
        )
    for slot_id, original_label in enumerate(spec.original_labels):
        ORIGINAL_TO_COARSE[original_label] = GROUP_COARSE_START + spec.group_id
        ORIGINAL_TO_GROUP[original_label] = spec.group_id
        ORIGINAL_TO_SLOT[original_label] = slot_id


def hierarchy_lookup_tensors(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return original-label lookup tables on ``device``."""
    return (
        torch.as_tensor(ORIGINAL_TO_COARSE, dtype=torch.long, device=device),
        torch.as_tensor(ORIGINAL_TO_GROUP, dtype=torch.long, device=device),
        torch.as_tensor(ORIGINAL_TO_SLOT, dtype=torch.long, device=device),
    )


def build_hierarchical_targets(
    segmentation: torch.Tensor,
    ignore_label: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert original CellMap labels to hierarchy targets.

    Returns ``(semantic, coarse, group, slot, valid_mask)``. Group and slot are
    ``-1`` for background/fixed voxels and are only consumed on dynamic voxels.
    """
    if segmentation.ndim < 2:
        raise ValueError("segmentation must have shape [B, 1, ...] or [B, ...]")
    if segmentation.shape[1] != 1:
        segmentation = segmentation[:, :1]

    semantic = segmentation.long()
    valid_mask = (semantic >= 0) & (semantic < NUM_ORIGINAL_CLASSES)
    if ignore_label is not None:
        valid_mask &= semantic != int(ignore_label)

    safe_semantic = semantic.clamp(min=0, max=NUM_ORIGINAL_CLASSES - 1)
    coarse_lookup, group_lookup, slot_lookup = hierarchy_lookup_tensors(semantic.device)
    coarse = coarse_lookup[safe_semantic]
    group = group_lookup[safe_semantic]
    slot = slot_lookup[safe_semantic]

    # Values under an invalid mask are inert but kept in legal ranges for CE.
    semantic = torch.where(valid_mask, safe_semantic, torch.zeros_like(safe_semantic))
    coarse = torch.where(valid_mask, coarse, torch.zeros_like(coarse))
    group = torch.where(valid_mask, group, torch.full_like(group, -1))
    slot = torch.where(valid_mask, slot, torch.full_like(slot, -1))
    return semantic, coarse, group, slot, valid_mask


def semantic_channel_layout() -> Tuple[Tuple[int, int], ...]:
    """Map every original label to ``(coarse_channel, slot_id)``."""
    return tuple(zip(ORIGINAL_TO_COARSE, ORIGINAL_TO_SLOT))
