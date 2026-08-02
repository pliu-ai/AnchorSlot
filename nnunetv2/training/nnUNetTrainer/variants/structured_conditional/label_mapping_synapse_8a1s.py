from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

# Fixed output head channels for the Synapse 13-organ AnchorSlot "8-anchor /
# 1-slot" variant: 8 fixed anchors (liver, spleen, stomach, aorta, ivc,
# pancreas, r_kidney, l_kidney) + a SINGLE dynamic conditional slot. Every
# dynamic group holds exactly one organ, so only COND_SLOT_1 is ever used and
# the active-slot mask has width 1. Head width = 11 (bg + 8 anchors + 1 slot +
# other). The network/loss/metrics read these constants, so the layout drives
# everything downstream.
BACKGROUND_CHANNEL = 0
LIVER_CHANNEL = 1
SPLEEN_CHANNEL = 2
STOMACH_CHANNEL = 3
AORTA_CHANNEL = 4
IVC_CHANNEL = 5
PANCREAS_CHANNEL = 6
RKIDNEY_CHANNEL = 7
LKIDNEY_CHANNEL = 8
COND_SLOT_1_CHANNEL = 9
OTHER_CHANNEL = 10
NUM_OUTPUT_CHANNELS = 11
NUM_DYNAMIC_GROUPS = 5
# Single conditional slot: every dynamic group has exactly one member.
MAX_CONDITIONAL_SLOTS = 1
# Original Synapse/BTCV label space = background + 13 organs.
NUM_ORIGINAL_LABELS = 14

OUTPUT_CHANNEL_NAMES: Tuple[str, ...] = (
    "background",
    "liver",
    "spleen",
    "stomach",
    "aorta",
    "ivc",
    "pancreas",
    "r_kidney",
    "l_kidney",
    "cond_slot_1",
    "other",
)

# Original Synapse/BTCV label IDs for fixed anchor classes.
FIXED_ORIGINAL_TO_OUTPUT: Dict[int, int] = {
    6: LIVER_CHANNEL,
    1: SPLEEN_CHANNEL,
    7: STOMACH_CHANNEL,
    8: AORTA_CHANNEL,
    9: IVC_CHANNEL,
    11: PANCREAS_CHANNEL,
    2: RKIDNEY_CHANNEL,
    3: LKIDNEY_CHANNEL,
}

FIXED_OUTPUT_TO_ORIGINAL: Dict[int, int] = {v: k for k, v in FIXED_ORIGINAL_TO_OUTPUT.items()}


@dataclass(frozen=True)
class DynamicGroupSpec:
    """Describes one dynamic group and its ordered conditional subclasses."""

    group_id: int
    short_name: str
    display_name: str
    original_labels: Tuple[int, ...]
    subclass_names: Tuple[str, ...]

    @property
    def num_slots(self) -> int:
        return len(self.original_labels)


# Synapse/BTCV dynamic groups for the 8-anchor / 1-slot variant. Anchors
# {liver=6, spleen=1, stomach=7, aorta=8, ivc=9, pancreas=11, r_kidney=2,
# l_kidney=3} are fixed; the remaining 5 organs each form their own single-slot
# dynamic group (gallbladder is dynamic here, unlike the anchor variants).
DYNAMIC_GROUP_SPECS: Tuple[DynamicGroupSpec, ...] = (
    DynamicGroupSpec(0, "G1", "Gallbladder", (4,), ("gallbladder",)),
    DynamicGroupSpec(1, "G2", "Esophagus", (5,), ("esophagus",)),
    DynamicGroupSpec(2, "G3", "PortalSplenicVein", (10,), ("portal_splenic_vein",)),
    DynamicGroupSpec(3, "G4", "RAdrenal", (12,), ("r_adrenal",)),
    DynamicGroupSpec(4, "G5", "LAdrenal", (13,), ("l_adrenal",)),
)

GROUP_ID_TO_SPEC: Dict[int, DynamicGroupSpec] = {spec.group_id: spec for spec in DYNAMIC_GROUP_SPECS}
ORIGINAL_LABEL_TO_GROUP_ID: Dict[int, int] = {
    original_label: spec.group_id
    for spec in DYNAMIC_GROUP_SPECS
    for original_label in spec.original_labels
}


def get_group_spec(group_id: int) -> DynamicGroupSpec:
    if int(group_id) not in GROUP_ID_TO_SPEC:
        raise ValueError(f"Unknown dynamic group_id={group_id}. Expected range [0, {NUM_DYNAMIC_GROUPS - 1}].")
    return GROUP_ID_TO_SPEC[int(group_id)]


def get_active_conditional_output_channels(group_id: int) -> Tuple[int, ...]:
    spec = get_group_spec(group_id)
    return tuple(COND_SLOT_1_CHANNEL + i for i in range(spec.num_slots))


def get_conditional_channel_to_original_label(group_id: int) -> Dict[int, int]:
    spec = get_group_spec(group_id)
    return {
        COND_SLOT_1_CHANNEL + i: int(original_label)
        for i, original_label in enumerate(spec.original_labels)
    }


def build_active_conditional_slot_mask(group_ids: torch.Tensor) -> torch.Tensor:
    """
    Build per-sample active slot mask.

    Returns:
        Tensor with shape [B, 1], where True means the conditional slot is active
        for the selected group and should participate in slot-specific losses/metrics.
        Every group is single-slot here, so the column is always True.
    """
    if group_ids.ndim != 1:
        group_ids = group_ids.reshape(-1)
    b = int(group_ids.shape[0])
    active = torch.zeros((b, MAX_CONDITIONAL_SLOTS), dtype=torch.bool, device=group_ids.device)
    for i in range(b):
        spec = get_group_spec(int(group_ids[i].item()))
        active[i, : spec.num_slots] = True
    return active


def infer_present_groups_from_segmentation(
    segmentation: torch.Tensor,
    ignore_label: Optional[int] = None,
) -> Set[int]:
    """Infer which dynamic groups are present in one segmentation tensor."""
    if segmentation.ndim > 0 and segmentation.shape[0] == 1:
        segmentation = segmentation[0]
    labels = torch.unique(segmentation).tolist()
    present: Set[int] = set()
    for label in labels:
        label_i = int(label)
        if label_i < 0:
            continue
        if ignore_label is not None and label_i == int(ignore_label):
            continue
        if label_i in ORIGINAL_LABEL_TO_GROUP_ID:
            present.add(int(ORIGINAL_LABEL_TO_GROUP_ID[label_i]))
    return present


def infer_present_groups_from_class_locations(properties: Mapping) -> Set[int]:
    """
    Infer dynamic group presence from nnUNet case properties['class_locations'].

    This is used by the dataloader to sample conditions at case level, and can be
    cached because properties are static per case.
    """
    class_locations = properties.get("class_locations", {})
    if not isinstance(class_locations, Mapping):
        return set()

    present: Set[int] = set()
    for spec in DYNAMIC_GROUP_SPECS:
        for original_label in spec.original_labels:
            coords = class_locations.get(int(original_label), None)
            if isinstance(coords, np.ndarray):
                if coords.size > 0:
                    present.add(spec.group_id)
                    break
            elif isinstance(coords, (list, tuple)):
                if len(coords) > 0:
                    present.add(spec.group_id)
                    break
    return present


def sample_group_id_for_case(
    present_group_ids: Sequence[int],
    p_present_group: float = 0.8,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Sample one dynamic group ID for one case.

    - With probability p_present_group, sample from present groups when possible.
    - Otherwise sample from absent groups when possible.
    """
    if rng is None:
        rng = np.random.default_rng()

    p_present_group = float(np.clip(p_present_group, 0.0, 1.0))
    present_sorted = sorted({int(i) for i in present_group_ids if 0 <= int(i) < NUM_DYNAMIC_GROUPS})
    absent = [i for i in range(NUM_DYNAMIC_GROUPS) if i not in present_sorted]

    if len(present_sorted) == 0:
        return int(rng.integers(0, NUM_DYNAMIC_GROUPS))

    draw_present = bool(rng.random() < p_present_group)
    if draw_present:
        return int(rng.choice(np.asarray(present_sorted, dtype=np.int64)))

    if len(absent) > 0:
        return int(rng.choice(np.asarray(absent, dtype=np.int64)))

    return int(rng.choice(np.asarray(present_sorted, dtype=np.int64)))


def remap_original_to_structured(
    segmentation: torch.Tensor,
    group_ids: torch.Tensor,
    ignore_label: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Remap original CellMap labels into the fixed 9-channel structured target space.

    Args:
        segmentation: [B, 1, ...] (or [B, ...]) tensor of original label IDs.
        group_ids: [B] dynamic group IDs in [0, 12].
        ignore_label: Optional dataset ignore label.

    Returns:
        remapped_target: [B, 1, ...], long, values in [0, 8] on valid voxels.
        valid_mask: [B, 1, ...], bool, False on ignored voxels.
        active_conditional_slots: [B, 3], bool, active slot mask.
    """
    if segmentation.ndim < 2:
        raise ValueError("segmentation must have shape [B, 1, ...] or [B, ...].")
    if segmentation.ndim >= 2 and segmentation.shape[1] != 1:
        segmentation = segmentation[:, :1]
    if group_ids.ndim != 1:
        group_ids = group_ids.reshape(-1)

    if segmentation.shape[0] != group_ids.shape[0]:
        raise ValueError(
            f"Batch size mismatch between segmentation ({segmentation.shape[0]}) and group_ids ({group_ids.shape[0]})."
        )

    seg = segmentation.long()
    remapped = torch.full_like(seg, fill_value=OTHER_CHANNEL, dtype=torch.long)
    invalid_mask = seg < 0
    if ignore_label is not None:
        invalid_mask = invalid_mask | (seg == int(ignore_label))
    valid_mask = ~invalid_mask

    # Background and fixed classes are globally defined.
    remapped[valid_mask & (seg == 0)] = BACKGROUND_CHANNEL
    for original_label, output_channel in FIXED_ORIGINAL_TO_OUTPUT.items():
        remapped[valid_mask & (seg == int(original_label))] = int(output_channel)

    # Group-dependent conditional slots.
    for b in range(seg.shape[0]):
        spec = get_group_spec(int(group_ids[b].item()))
        seg_b = seg[b, 0]
        valid_b = valid_mask[b, 0]
        remapped_b = remapped[b, 0]
        for slot_idx, original_label in enumerate(spec.original_labels):
            output_channel = COND_SLOT_1_CHANNEL + slot_idx
            remapped_b[valid_b & (seg_b == int(original_label))] = int(output_channel)

    # Invalid voxels are set to background but masked out by valid_mask.
    remapped[~valid_mask] = BACKGROUND_CHANNEL
    active_conditional_slots = build_active_conditional_slot_mask(group_ids)
    return remapped, valid_mask, active_conditional_slots


def structured_prediction_to_original_labels(
    structured_prediction: torch.Tensor,
    group_id: int,
    background_value: int = 0,
) -> torch.Tensor:
    """
    Convert one structured prediction map back to original CellMap label IDs.

    The `other` channel is intentionally not mapped to an original semantic class.
    It is mapped to `background_value` in the reconstructed output.
    """
    pred = structured_prediction.long()
    out = torch.full_like(pred, fill_value=int(background_value), dtype=torch.long)

    for output_channel, original_label in FIXED_OUTPUT_TO_ORIGINAL.items():
        out[pred == int(output_channel)] = int(original_label)

    cond_mapping = get_conditional_channel_to_original_label(group_id)
    for output_channel, original_label in cond_mapping.items():
        out[pred == int(output_channel)] = int(original_label)

    return out


def original_label_name_lookup() -> Dict[int, str]:
    """Optional helper for readable logs/reports."""
    names = {
        0: "background",
        1: "spleen",
        2: "r_kidney",
        3: "l_kidney",
        4: "gallbladder",
        5: "esophagus",
        6: "liver",
        7: "stomach",
        8: "aorta",
        9: "ivc",
        10: "portal_splenic_vein",
        11: "pancreas",
        12: "r_adrenal",
        13: "l_adrenal",
    }
    return names


def flatten_active_conditional_label_names() -> List[str]:
    """Returns all active dynamic subclass names in group order for reporting."""
    names: List[str] = []
    for spec in DYNAMIC_GROUP_SPECS:
        for slot_idx, subclass_name in enumerate(spec.subclass_names):
            names.append(f"{spec.short_name}:{subclass_name}:slot{slot_idx + 1}")
    return names


def iter_group_specs() -> Iterable[DynamicGroupSpec]:
    return iter(DYNAMIC_GROUP_SPECS)
