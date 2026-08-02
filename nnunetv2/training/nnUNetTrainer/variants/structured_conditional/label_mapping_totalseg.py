from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

# AnchorSlot label organization for TotalSegmentator v2 (117 structures).
# ------------------------------------------------------------------
# TotalSegmentator CTs cover variable body regions, so most structures are absent
# (out of field of view) in any given scan. Within the FOV the annotation is dense.
# 4 large, high-contrast, frequently-imaged organs are fixed anchors; the remaining
# 113 structures are organized into 44 anatomically-related dynamic groups of <=3
# members each, reusing the standard 9-channel structured head (bg + 4 anchors +
# 3 conditional slots + non-query "other"). Groups with <3 members mask unused slots.
BACKGROUND_CHANNEL = 0
LIVER_CHANNEL = 1
SPLEEN_CHANNEL = 2
AORTA_CHANNEL = 3
HEART_CHANNEL = 4
COND_SLOT_1_CHANNEL = 5
COND_SLOT_2_CHANNEL = 6
COND_SLOT_3_CHANNEL = 7
OTHER_CHANNEL = 8
NUM_OUTPUT_CHANNELS = 9
# TotalSegmentator v2 label space = background + 117 structures.
NUM_ORIGINAL_LABELS = 118

OUTPUT_CHANNEL_NAMES: Tuple[str, ...] = (
    "background", "liver", "spleen", "aorta", "heart",
    "cond_slot_1", "cond_slot_2", "cond_slot_3", "other",
)

# Original TotalSegmentator label IDs for the fixed anchors.
FIXED_ORIGINAL_TO_OUTPUT: Dict[int, int] = {5: LIVER_CHANNEL, 1: SPLEEN_CHANNEL, 52: AORTA_CHANNEL, 51: HEART_CHANNEL}
FIXED_OUTPUT_TO_ORIGINAL: Dict[int, int] = {v: k for k, v in FIXED_ORIGINAL_TO_OUTPUT.items()}

# Full v2 name table (label id -> name).
_NAMES: Dict[int, str] = {
    0: "background", 1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder", 5: "liver",
    6: "stomach", 7: "pancreas", 8: "adrenal_gland_right", 9: "adrenal_gland_left",
    10: "lung_upper_lobe_left", 11: "lung_lower_lobe_left", 12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right", 14: "lung_lower_lobe_right", 15: "esophagus", 16: "trachea",
    17: "thyroid_gland", 18: "small_bowel", 19: "duodenum", 20: "colon", 21: "urinary_bladder",
    22: "prostate", 23: "kidney_cyst_left", 24: "kidney_cyst_right", 25: "sacrum",
    26: "vertebrae_S1", 27: "vertebrae_L5", 28: "vertebrae_L4", 29: "vertebrae_L3", 30: "vertebrae_L2",
    31: "vertebrae_L1", 32: "vertebrae_T12", 33: "vertebrae_T11", 34: "vertebrae_T10", 35: "vertebrae_T9",
    36: "vertebrae_T8", 37: "vertebrae_T7", 38: "vertebrae_T6", 39: "vertebrae_T5", 40: "vertebrae_T4",
    41: "vertebrae_T3", 42: "vertebrae_T2", 43: "vertebrae_T1", 44: "vertebrae_C7", 45: "vertebrae_C6",
    46: "vertebrae_C5", 47: "vertebrae_C4", 48: "vertebrae_C3", 49: "vertebrae_C2", 50: "vertebrae_C1",
    51: "heart", 52: "aorta", 53: "pulmonary_vein", 54: "brachiocephalic_trunk",
    55: "subclavian_artery_right", 56: "subclavian_artery_left", 57: "common_carotid_artery_right",
    58: "common_carotid_artery_left", 59: "brachiocephalic_vein_left", 60: "brachiocephalic_vein_right",
    61: "atrial_appendage_left", 62: "superior_vena_cava", 63: "inferior_vena_cava",
    64: "portal_vein_and_splenic_vein", 65: "iliac_artery_left", 66: "iliac_artery_right",
    67: "iliac_vena_left", 68: "iliac_vena_right", 69: "humerus_left", 70: "humerus_right",
    71: "scapula_left", 72: "scapula_right", 73: "clavicula_left", 74: "clavicula_right",
    75: "femur_left", 76: "femur_right", 77: "hip_left", 78: "hip_right", 79: "spinal_cord",
    80: "gluteus_maximus_left", 81: "gluteus_maximus_right", 82: "gluteus_medius_left",
    83: "gluteus_medius_right", 84: "gluteus_minimus_left", 85: "gluteus_minimus_right",
    86: "autochthon_left", 87: "autochthon_right", 88: "iliopsoas_left", 89: "iliopsoas_right",
    90: "brain", 91: "skull",
    **{91 + i: f"rib_left_{i}" for i in range(1, 13)},      # 92..103
    **{103 + i: f"rib_right_{i}" for i in range(1, 13)},    # 104..115
    116: "sternum", 117: "costal_cartilages",
}

# (short_name, display_name, original_label_ids) for the 44 dynamic groups.
_GROUP_DEFS: Tuple[Tuple[str, str, Tuple[int, ...]], ...] = (
    ("G1", "Kidneys", (2, 3)),
    ("G2", "KidneyCysts", (23, 24)),
    ("G3", "Adrenals", (8, 9)),
    ("G4", "GallbladderPancreas", (4, 7)),
    ("G5", "UpperGI", (6, 15, 19)),
    ("G6", "LowerGI", (18, 20)),
    ("G7", "PelvicOrgans", (21, 22)),
    ("G8", "Airway", (16, 17)),
    ("G9", "LungLeft", (10, 11)),
    ("G10", "LungRight", (12, 13, 14)),
    ("G11", "SacrumS1", (25, 26)),
    ("G12", "LumbarLow", (27, 28, 29)),
    ("G13", "LumbarUp", (30, 31, 32)),
    ("G14", "ThoracicT9_11", (33, 34, 35)),
    ("G15", "ThoracicT6_8", (36, 37, 38)),
    ("G16", "ThoracicT3_5", (39, 40, 41)),
    ("G17", "ThoracicT1_2_C7", (42, 43, 44)),
    ("G18", "CervicalC4_6", (45, 46, 47)),
    ("G19", "CervicalC1_3", (48, 49, 50)),
    ("G20", "ArchArteries", (54, 55, 56)),
    ("G21", "CarotidsPulmVein", (57, 58, 53)),
    ("G22", "ThoracicVeins", (59, 60, 62)),
    ("G23", "CentralVeins", (63, 64, 61)),
    ("G24", "IliacArteries", (65, 66)),
    ("G25", "IliacVeins", (67, 68)),
    ("G26", "UpperLimbBones1", (69, 70, 71)),
    ("G27", "UpperLimbBones2", (72, 73, 74)),
    ("G28", "PelvisBones", (77, 78)),
    ("G29", "Femurs", (75, 76)),
    ("G30", "CNS", (79, 90, 91)),
    ("G31", "GluteusMax", (80, 81)),
    ("G32", "GluteusMed", (82, 83)),
    ("G33", "GluteusMin", (84, 85)),
    ("G34", "Autochthon", (86, 87)),
    ("G35", "Iliopsoas", (88, 89)),
    ("G36", "RibsL1_3", (92, 93, 94)),
    ("G37", "RibsL4_6", (95, 96, 97)),
    ("G38", "RibsL7_9", (98, 99, 100)),
    ("G39", "RibsL10_12", (101, 102, 103)),
    ("G40", "RibsR1_3", (104, 105, 106)),
    ("G41", "RibsR4_6", (107, 108, 109)),
    ("G42", "RibsR7_9", (110, 111, 112)),
    ("G43", "RibsR10_12", (113, 114, 115)),
    ("G44", "SternumCartilage", (116, 117)),
)
NUM_DYNAMIC_GROUPS = len(_GROUP_DEFS)


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


DYNAMIC_GROUP_SPECS: Tuple[DynamicGroupSpec, ...] = tuple(
    DynamicGroupSpec(i, short, name, labels, tuple(_NAMES[l] for l in labels))
    for i, (short, name, labels) in enumerate(_GROUP_DEFS)
)

GROUP_ID_TO_SPEC: Dict[int, DynamicGroupSpec] = {s.group_id: s for s in DYNAMIC_GROUP_SPECS}
ORIGINAL_LABEL_TO_GROUP_ID: Dict[int, int] = {
    l: s.group_id for s in DYNAMIC_GROUP_SPECS for l in s.original_labels
}

# Sanity: every non-anchor, non-background label belongs to exactly one group.
_assigned = set(FIXED_ORIGINAL_TO_OUTPUT) | set(ORIGINAL_LABEL_TO_GROUP_ID)
assert _assigned == set(range(1, NUM_ORIGINAL_LABELS)), (
    f"label coverage mismatch: missing={set(range(1, NUM_ORIGINAL_LABELS)) - _assigned}, "
    f"dup_or_extra={_assigned - set(range(1, NUM_ORIGINAL_LABELS))}"
)
assert max(s.num_slots for s in DYNAMIC_GROUP_SPECS) <= 3


def get_group_spec(group_id: int) -> DynamicGroupSpec:
    if int(group_id) not in GROUP_ID_TO_SPEC:
        raise ValueError(f"Unknown dynamic group_id={group_id}. Expected [0,{NUM_DYNAMIC_GROUPS - 1}].")
    return GROUP_ID_TO_SPEC[int(group_id)]


def get_active_conditional_output_channels(group_id: int) -> Tuple[int, ...]:
    return tuple(COND_SLOT_1_CHANNEL + i for i in range(get_group_spec(group_id).num_slots))


def get_conditional_channel_to_original_label(group_id: int) -> Dict[int, int]:
    spec = get_group_spec(group_id)
    return {COND_SLOT_1_CHANNEL + i: int(l) for i, l in enumerate(spec.original_labels)}


def build_active_conditional_slot_mask(group_ids: torch.Tensor) -> torch.Tensor:
    if group_ids.ndim != 1:
        group_ids = group_ids.reshape(-1)
    b = int(group_ids.shape[0])
    active = torch.zeros((b, 3), dtype=torch.bool, device=group_ids.device)
    for i in range(b):
        active[i, : get_group_spec(int(group_ids[i].item())).num_slots] = True
    return active


def infer_present_groups_from_segmentation(segmentation: torch.Tensor, ignore_label: Optional[int] = None) -> Set[int]:
    if segmentation.ndim > 0 and segmentation.shape[0] == 1:
        segmentation = segmentation[0]
    present: Set[int] = set()
    for label in torch.unique(segmentation).tolist():
        li = int(label)
        if li < 0 or (ignore_label is not None and li == int(ignore_label)):
            continue
        if li in ORIGINAL_LABEL_TO_GROUP_ID:
            present.add(int(ORIGINAL_LABEL_TO_GROUP_ID[li]))
    return present


def infer_present_groups_from_class_locations(properties: Mapping) -> Set[int]:
    class_locations = properties.get("class_locations", {})
    if not isinstance(class_locations, Mapping):
        return set()
    present: Set[int] = set()
    for spec in DYNAMIC_GROUP_SPECS:
        for l in spec.original_labels:
            coords = class_locations.get(int(l), None)
            if isinstance(coords, np.ndarray) and coords.size > 0:
                present.add(spec.group_id); break
            if isinstance(coords, (list, tuple)) and len(coords) > 0:
                present.add(spec.group_id); break
    return present


def sample_group_id_for_case(present_group_ids: Sequence[int], p_present_group: float = 0.8,
                             rng: Optional[np.random.Generator] = None) -> int:
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


def remap_original_to_structured(segmentation: torch.Tensor, group_ids: torch.Tensor,
                                 ignore_label: Optional[int] = None):
    if segmentation.ndim >= 2 and segmentation.shape[1] != 1:
        segmentation = segmentation[:, :1]
    if group_ids.ndim != 1:
        group_ids = group_ids.reshape(-1)
    seg = segmentation.long()
    remapped = torch.full_like(seg, fill_value=OTHER_CHANNEL, dtype=torch.long)
    invalid = seg < 0
    if ignore_label is not None:
        invalid = invalid | (seg == int(ignore_label))
    valid = ~invalid
    remapped[valid & (seg == 0)] = BACKGROUND_CHANNEL
    for original_label, out_ch in FIXED_ORIGINAL_TO_OUTPUT.items():
        remapped[valid & (seg == int(original_label))] = int(out_ch)
    for b in range(seg.shape[0]):
        spec = get_group_spec(int(group_ids[b].item()))
        seg_b, valid_b, rem_b = seg[b, 0], valid[b, 0], remapped[b, 0]
        for slot_idx, l in enumerate(spec.original_labels):
            rem_b[valid_b & (seg_b == int(l))] = int(COND_SLOT_1_CHANNEL + slot_idx)
    remapped[~valid] = BACKGROUND_CHANNEL
    return remapped, valid, build_active_conditional_slot_mask(group_ids)


def structured_prediction_to_original_labels(structured_prediction: torch.Tensor, group_id: int,
                                             background_value: int = 0) -> torch.Tensor:
    pred = structured_prediction.long()
    out = torch.full_like(pred, fill_value=int(background_value), dtype=torch.long)
    for out_ch, original_label in FIXED_OUTPUT_TO_ORIGINAL.items():
        out[pred == int(out_ch)] = int(original_label)
    for out_ch, original_label in get_conditional_channel_to_original_label(group_id).items():
        out[pred == int(out_ch)] = int(original_label)
    return out


def original_label_name_lookup() -> Dict[int, str]:
    return dict(_NAMES)


def flatten_active_conditional_label_names() -> List[str]:
    names: List[str] = []
    for spec in DYNAMIC_GROUP_SPECS:
        for slot_idx, sub in enumerate(spec.subclass_names):
            names.append(f"{spec.short_name}:{sub}:slot{slot_idx + 1}")
    return names


def iter_group_specs() -> Iterable[DynamicGroupSpec]:
    return iter(DYNAMIC_GROUP_SPECS)
