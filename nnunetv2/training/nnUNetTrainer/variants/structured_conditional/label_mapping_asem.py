from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

# ASEM all-conditional AnchorSlot variant.
# ------------------------------------------------------------------
# ASEM is a pure partial-label dataset: every annotated crop is binary for ONE
# structure (mito/er/golgi/ccp/np), and NO crop co-labels multiple structures.
# There is therefore no usable fixed anchor (an anchor would receive false-negative
# background on crops of other structures). The MAIN instantiation uses ZERO fixed
# anchors -> every structure is a single-slot dynamic condition.
#
# To reuse the loss/network/metrics/inference code verbatim we keep the same
# physical head layout as the CellMap/Synapse variants (3 conditional slots + an
# `other` channel), but with no anchor channels. Every ASEM group has exactly one
# slot, so COND_SLOT_2/3 are always masked by the active-slot mask.
BACKGROUND_CHANNEL = 0
COND_SLOT_1_CHANNEL = 1
COND_SLOT_2_CHANNEL = 2
COND_SLOT_3_CHANNEL = 3
OTHER_CHANNEL = 4
NUM_OUTPUT_CHANNELS = 5
NUM_DYNAMIC_GROUPS = 5
# Original ASEM label space = background + 5 structures.
NUM_ORIGINAL_LABELS = 6

OUTPUT_CHANNEL_NAMES: Tuple[str, ...] = (
    "background",
    "cond_slot_1",
    "cond_slot_2",
    "cond_slot_3",
    "other",
)

# No fixed anchors in the all-conditional ASEM variant.
FIXED_ORIGINAL_TO_OUTPUT: Dict[int, int] = {}
FIXED_OUTPUT_TO_ORIGINAL: Dict[int, int] = {}


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


# Original ASEM label IDs (set in extract_asem_dataset.py):
# mito=1, er=2, golgi=3, ccp=4, np=5. Each structure is its own single-slot group.
DYNAMIC_GROUP_SPECS: Tuple[DynamicGroupSpec, ...] = (
    DynamicGroupSpec(0, "G1", "Mito", (1,), ("mito",)),
    DynamicGroupSpec(1, "G2", "ER", (2,), ("er",)),
    DynamicGroupSpec(2, "G3", "Golgi", (3,), ("golgi",)),
    DynamicGroupSpec(3, "G4", "CCP", (4,), ("ccp",)),
    DynamicGroupSpec(4, "G5", "NP", (5,), ("np",)),
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
    """Per-sample active slot mask of shape [B, 3]; only slot 1 is ever active for ASEM."""
    if group_ids.ndim != 1:
        group_ids = group_ids.reshape(-1)
    b = int(group_ids.shape[0])
    active = torch.zeros((b, 3), dtype=torch.bool, device=group_ids.device)
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
    """Infer dynamic group presence from nnUNet case properties['class_locations']."""
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
    p_present_group: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Sample one dynamic group ID for one case.

    For ASEM the default p_present_group=1.0: only ever condition on the structure
    that is actually annotated in the crop. Conditioning on an absent structure
    would supervise it as all-background, which is wrong under partial labels (the
    structure may be physically present but unlabeled in this crop).
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
    Remap original ASEM labels into the fixed 5-channel structured target space.

    Returns:
        remapped_target: [B, 1, ...], long, values in [0, 4] on valid voxels.
        valid_mask: [B, 1, ...], bool, False on ignored voxels.
        active_conditional_slots: [B, 3], bool, active slot mask (only slot 1 active).
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

    # Background is globally defined; there are no fixed anchors.
    remapped[valid_mask & (seg == 0)] = BACKGROUND_CHANNEL

    # Group-dependent conditional slots.
    for b in range(seg.shape[0]):
        spec = get_group_spec(int(group_ids[b].item()))
        seg_b = seg[b, 0]
        valid_b = valid_mask[b, 0]
        remapped_b = remapped[b, 0]
        for slot_idx, original_label in enumerate(spec.original_labels):
            output_channel = COND_SLOT_1_CHANNEL + slot_idx
            remapped_b[valid_b & (seg_b == int(original_label))] = int(output_channel)

    remapped[~valid_mask] = BACKGROUND_CHANNEL
    active_conditional_slots = build_active_conditional_slot_mask(group_ids)
    return remapped, valid_mask, active_conditional_slots


def structured_prediction_to_original_labels(
    structured_prediction: torch.Tensor,
    group_id: int,
    background_value: int = 0,
) -> torch.Tensor:
    """Convert one structured prediction map back to original ASEM label IDs."""
    pred = structured_prediction.long()
    out = torch.full_like(pred, fill_value=int(background_value), dtype=torch.long)

    for output_channel, original_label in FIXED_OUTPUT_TO_ORIGINAL.items():
        out[pred == int(output_channel)] = int(original_label)

    cond_mapping = get_conditional_channel_to_original_label(group_id)
    for output_channel, original_label in cond_mapping.items():
        out[pred == int(output_channel)] = int(original_label)

    return out


def original_label_name_lookup() -> Dict[int, str]:
    return {0: "background", 1: "mito", 2: "er", 3: "golgi", 4: "ccp", 5: "np"}


def flatten_active_conditional_label_names() -> List[str]:
    names: List[str] = []
    for spec in DYNAMIC_GROUP_SPECS:
        for slot_idx, subclass_name in enumerate(spec.subclass_names):
            names.append(f"{spec.short_name}:{subclass_name}:slot{slot_idx + 1}")
    return names


def iter_group_specs() -> Iterable[DynamicGroupSpec]:
    return iter(DYNAMIC_GROUP_SPECS)
