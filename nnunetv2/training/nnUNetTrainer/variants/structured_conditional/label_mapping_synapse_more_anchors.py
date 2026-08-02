from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

BACKGROUND_CHANNEL = 0
LIVER_CHANNEL = 1
SPLEEN_CHANNEL = 2
STOMACH_CHANNEL = 3
AORTA_CHANNEL = 4
R_KIDNEY_CHANNEL = 5
L_KIDNEY_CHANNEL = 6
IVC_CHANNEL = 7
PANCREAS_CHANNEL = 8
COND_SLOT_1_CHANNEL = 9
COND_SLOT_2_CHANNEL = 10
OTHER_CHANNEL = 11
NUM_OUTPUT_CHANNELS = 12
NUM_COND_SLOTS = 2
NUM_DYNAMIC_GROUPS = 3
NUM_ORIGINAL_LABELS = 14

OUTPUT_CHANNEL_NAMES: Tuple[str, ...] = (
    "background",
    "liver",
    "spleen",
    "stomach",
    "aorta",
    "r_kidney",
    "l_kidney",
    "ivc",
    "pancreas",
    "cond_slot_1",
    "cond_slot_2",
    "other",
)

FIXED_ORIGINAL_TO_OUTPUT: Dict[int, int] = {
    6: LIVER_CHANNEL,
    1: SPLEEN_CHANNEL,
    7: STOMACH_CHANNEL,
    8: AORTA_CHANNEL,
    2: R_KIDNEY_CHANNEL,
    3: L_KIDNEY_CHANNEL,
    9: IVC_CHANNEL,
    11: PANCREAS_CHANNEL,
}

FIXED_OUTPUT_TO_ORIGINAL: Dict[int, int] = {
    LIVER_CHANNEL: 6,
    SPLEEN_CHANNEL: 1,
    STOMACH_CHANNEL: 7,
    AORTA_CHANNEL: 8,
    R_KIDNEY_CHANNEL: 2,
    L_KIDNEY_CHANNEL: 3,
    IVC_CHANNEL: 9,
    PANCREAS_CHANNEL: 11,
}


@dataclass(frozen=True)
class DynamicGroupSpec:
    group_id: int
    short_name: str
    display_name: str
    original_labels: Tuple[int, ...]
    subclass_names: Tuple[str, ...]

    @property
    def num_slots(self) -> int:
        return len(self.original_labels)


DYNAMIC_GROUP_SPECS: Tuple[DynamicGroupSpec, ...] = (
    DynamicGroupSpec(0, "G1", "Adrenals", (12, 13), ("r_adrenal", "l_adrenal")),
    DynamicGroupSpec(1, "G2", "PortalGallbladder", (10, 4), ("portal_splenic_vein", "gallbladder")),
    DynamicGroupSpec(2, "G3", "Esophagus", (5,), ("esophagus",)),
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
    if group_ids.ndim != 1:
        group_ids = group_ids.reshape(-1)
    active = torch.zeros((int(group_ids.shape[0]), NUM_COND_SLOTS), dtype=torch.bool, device=group_ids.device)
    for i in range(int(group_ids.shape[0])):
        spec = get_group_spec(int(group_ids[i].item()))
        active[i, : spec.num_slots] = True
    return active


def infer_present_groups_from_segmentation(segmentation: torch.Tensor, ignore_label: Optional[int] = None) -> Set[int]:
    if segmentation.ndim > 0 and segmentation.shape[0] == 1:
        segmentation = segmentation[0]
    present: Set[int] = set()
    for label in torch.unique(segmentation).tolist():
        label_i = int(label)
        if label_i < 0 or (ignore_label is not None and label_i == int(ignore_label)):
            continue
        if label_i in ORIGINAL_LABEL_TO_GROUP_ID:
            present.add(int(ORIGINAL_LABEL_TO_GROUP_ID[label_i]))
    return present


def infer_present_groups_from_class_locations(properties: Mapping) -> Set[int]:
    class_locations = properties.get("class_locations", {})
    if not isinstance(class_locations, Mapping):
        return set()

    present: Set[int] = set()
    for spec in DYNAMIC_GROUP_SPECS:
        for original_label in spec.original_labels:
            coords = class_locations.get(int(original_label), None)
            if isinstance(coords, np.ndarray) and coords.size > 0:
                present.add(spec.group_id)
                break
            if isinstance(coords, (list, tuple)) and len(coords) > 0:
                present.add(spec.group_id)
                break
    return present


def sample_group_id_for_case(
    present_group_ids: Sequence[int],
    p_present_group: float = 0.8,
    rng: Optional[np.random.Generator] = None,
) -> int:
    if rng is None:
        rng = np.random.default_rng()

    p_present_group = float(np.clip(p_present_group, 0.0, 1.0))
    present_sorted = sorted({int(i) for i in present_group_ids if 0 <= int(i) < NUM_DYNAMIC_GROUPS})
    absent = [i for i in range(NUM_DYNAMIC_GROUPS) if i not in present_sorted]

    if len(present_sorted) == 0:
        return int(rng.integers(0, NUM_DYNAMIC_GROUPS))
    if bool(rng.random() < p_present_group):
        return int(rng.choice(np.asarray(present_sorted, dtype=np.int64)))
    if len(absent) > 0:
        return int(rng.choice(np.asarray(absent, dtype=np.int64)))
    return int(rng.choice(np.asarray(present_sorted, dtype=np.int64)))


def remap_original_to_structured(
    segmentation: torch.Tensor,
    group_ids: torch.Tensor,
    ignore_label: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if segmentation.ndim < 2:
        raise ValueError("segmentation must have shape [B, 1, ...] or [B, ...].")
    if segmentation.ndim >= 2 and segmentation.shape[1] != 1:
        segmentation = segmentation[:, :1]
    if group_ids.ndim != 1:
        group_ids = group_ids.reshape(-1)
    if segmentation.shape[0] != group_ids.shape[0]:
        raise ValueError(f"Batch size mismatch: segmentation={segmentation.shape[0]}, group_ids={group_ids.shape[0]}")

    seg = segmentation.long()
    remapped = torch.full_like(seg, fill_value=OTHER_CHANNEL, dtype=torch.long)
    invalid_mask = seg < 0
    if ignore_label is not None:
        invalid_mask = invalid_mask | (seg == int(ignore_label))
    valid_mask = ~invalid_mask

    remapped[valid_mask & (seg == 0)] = BACKGROUND_CHANNEL
    for original_label, output_channel in FIXED_ORIGINAL_TO_OUTPUT.items():
        remapped[valid_mask & (seg == int(original_label))] = int(output_channel)

    for b in range(seg.shape[0]):
        spec = get_group_spec(int(group_ids[b].item()))
        seg_b = seg[b, 0]
        valid_b = valid_mask[b, 0]
        remapped_b = remapped[b, 0]
        for slot_idx, original_label in enumerate(spec.original_labels):
            remapped_b[valid_b & (seg_b == int(original_label))] = int(COND_SLOT_1_CHANNEL + slot_idx)

    remapped[~valid_mask] = BACKGROUND_CHANNEL
    return remapped, valid_mask, build_active_conditional_slot_mask(group_ids)


def structured_prediction_to_original_labels(
    structured_prediction: torch.Tensor,
    group_id: int,
    background_value: int = 0,
) -> torch.Tensor:
    pred = structured_prediction.long()
    out = torch.full_like(pred, fill_value=int(background_value), dtype=torch.long)
    for output_channel, original_label in FIXED_OUTPUT_TO_ORIGINAL.items():
        out[pred == int(output_channel)] = int(original_label)
    for output_channel, original_label in get_conditional_channel_to_original_label(group_id).items():
        out[pred == int(output_channel)] = int(original_label)
    return out


def original_label_name_lookup() -> Dict[int, str]:
    return {
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


def flatten_active_conditional_label_names() -> List[str]:
    names: List[str] = []
    for spec in DYNAMIC_GROUP_SPECS:
        for slot_idx, subclass_name in enumerate(spec.subclass_names):
            names.append(f"{spec.short_name}:{subclass_name}:slot{slot_idx + 1}")
    return names


def iter_group_specs() -> Iterable[DynamicGroupSpec]:
    return iter(DYNAMIC_GROUP_SPECS)
