"""
Grouped-softmax flat head: the AnchorSlot target code without conditional computation.

One unconditioned network predicts 45 logits per voxel:
    0 background | 1-4 anchors (ecs, pm, cyto, nucpl) | 5-31 the 27 dynamic labels | 32-44 one
    non-query "other" logit per dynamic group.
For group g, the usual 9-channel structured logits are *gathered* from these (background, anchors,
g's labels in slot order, a constant INACTIVE_SLOT_LOGIT for an unused slot, other_g), so the same
StructuredConditionalLoss, validation metrics, predictor and reconstruction apply unchanged.
Training supervises all 13 group softmaxes every step instead of one sampled query.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .label_mapping_er_dynamic import (
    BACKGROUND_CHANNEL,
    COND_SLOT_1_CHANNEL,
    DYNAMIC_GROUP_SPECS,
    FIXED_ORIGINAL_TO_OUTPUT,
    FIXED_OUTPUT_TO_ORIGINAL,
    NUM_DYNAMIC_GROUPS,
    NUM_OUTPUT_CHANNELS,
    OTHER_CHANNEL,
)

INACTIVE_SLOT_LOGIT = -1e4  # same constant the structured CE uses for unused slots

_DYNAMIC_LABELS_IN_ORDER: Tuple[int, ...] = tuple(
    int(label) for spec in DYNAMIC_GROUP_SPECS for label in spec.original_labels
)
FLAT_CHANNEL_OF_ORIGINAL_LABEL: Dict[int, int] = {
    0: BACKGROUND_CHANNEL,
    **{int(original): int(channel) for channel, original in FIXED_OUTPUT_TO_ORIGINAL.items()},
    **{label: 1 + len(FIXED_OUTPUT_TO_ORIGINAL) + i for i, label in enumerate(_DYNAMIC_LABELS_IN_ORDER)},
}
_FIRST_OTHER_FLAT_CHANNEL = len(FLAT_CHANNEL_OF_ORIGINAL_LABEL)
OTHER_FLAT_CHANNEL: Dict[int, int] = {g: _FIRST_OTHER_FLAT_CHANNEL + g for g in range(NUM_DYNAMIC_GROUPS)}
NUM_FLAT_OUTPUT_CHANNELS = _FIRST_OTHER_FLAT_CHANNEL + NUM_DYNAMIC_GROUPS


def _build_gather_table() -> torch.Tensor:
    """[num_groups, 9] flat-channel index per structured channel; -1 marks an unused slot."""
    table = torch.full((NUM_DYNAMIC_GROUPS, NUM_OUTPUT_CHANNELS), -1, dtype=torch.long)
    for spec in DYNAMIC_GROUP_SPECS:
        g = int(spec.group_id)
        table[g, BACKGROUND_CHANNEL] = FLAT_CHANNEL_OF_ORIGINAL_LABEL[0]
        for channel, original in FIXED_OUTPUT_TO_ORIGINAL.items():
            table[g, int(channel)] = FLAT_CHANNEL_OF_ORIGINAL_LABEL[int(original)]
        for k, original in enumerate(spec.original_labels):
            table[g, COND_SLOT_1_CHANNEL + k] = FLAT_CHANNEL_OF_ORIGINAL_LABEL[int(original)]
        table[g, OTHER_CHANNEL] = OTHER_FLAT_CHANNEL[g]
    return table


_GATHER_TABLE = _build_gather_table()


def gather_group_logits(flat_logits: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    """[B, 45, ...] flat logits + [B] group ids -> [B, 9, ...] structured logits of each sample's group."""
    b = int(flat_logits.shape[0])
    spatial = flat_logits.shape[2:]
    index = _GATHER_TABLE.to(flat_logits.device)[group_ids.reshape(-1).long().to(flat_logits.device)]
    if index.shape[0] != b:
        raise ValueError(f"group_ids has {index.shape[0]} entries for a batch of {b}")
    unused = index < 0
    index_full = index.clamp_min(0).view(b, NUM_OUTPUT_CHANNELS, *([1] * len(spatial))).expand(
        b, NUM_OUTPUT_CHANNELS, *spatial
    )
    gathered = torch.gather(flat_logits, 1, index_full)
    fill = unused.view(b, NUM_OUTPUT_CHANNELS, *([1] * len(spatial))).expand_as(gathered)
    return gathered.masked_fill(fill, INACTIVE_SLOT_LOGIT)


# Table-based equivalent of label_mapping_er_dynamic.remap_original_to_structured. The reference uses
# per-label boolean-index assignments (a device sync each); the grouped loss remaps 13 groups x every
# deep-supervision level per step, so it uses one gather instead. Index = original label + 1, with
# -1 (padding) -> 0 and every id > 31 -> the last column, which maps to `other` like the reference.
_REMAP_TABLE_SIZE = 34


def _build_remap_table() -> torch.Tensor:
    table = torch.full((NUM_DYNAMIC_GROUPS, _REMAP_TABLE_SIZE), OTHER_CHANNEL, dtype=torch.long)
    table[:, 0] = BACKGROUND_CHANNEL
    table[:, 1] = BACKGROUND_CHANNEL
    for original, channel in FIXED_ORIGINAL_TO_OUTPUT.items():
        table[:, int(original) + 1] = int(channel)
    for spec in DYNAMIC_GROUP_SPECS:
        for k, original in enumerate(spec.original_labels):
            table[int(spec.group_id), int(original) + 1] = COND_SLOT_1_CHANNEL + k
    return table


def _build_active_slot_table() -> torch.Tensor:
    table = torch.zeros((NUM_DYNAMIC_GROUPS, 3), dtype=torch.bool)
    for spec in DYNAMIC_GROUP_SPECS:
        table[int(spec.group_id), : spec.num_slots] = True
    return table


_REMAP_TABLE = _build_remap_table()
_ACTIVE_SLOT_TABLE = _build_active_slot_table()


def remap_original_to_structured_fast(
    segmentation: torch.Tensor,
    group_ids: torch.Tensor,
    ignore_label: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same outputs as remap_original_to_structured (remapped, valid_mask, active_slots), sync-free."""
    if segmentation.ndim < 2:
        raise ValueError("segmentation must have shape [B, 1, ...] or [B, ...].")
    seg = segmentation.long()
    if seg.shape[1] != 1:
        seg = seg[:, :1]
    group_ids = group_ids.reshape(-1).long()
    if seg.shape[0] != group_ids.shape[0]:
        raise ValueError(
            f"Batch size mismatch between segmentation ({seg.shape[0]}) and group_ids ({group_ids.shape[0]})."
        )
    invalid = seg < 0
    if ignore_label is not None:
        invalid = invalid | (seg == int(ignore_label))
    index = (seg.clamp(-1, _REMAP_TABLE_SIZE - 2) + 1).reshape(seg.shape[0], -1)
    per_sample_table = _REMAP_TABLE.to(seg.device)[group_ids.to(seg.device)]
    remapped = torch.gather(per_sample_table, 1, index).view_as(seg).masked_fill(invalid, BACKGROUND_CHANNEL)
    active = _ACTIVE_SLOT_TABLE.to(group_ids.device)[group_ids]
    return remapped, ~invalid, active


def remap_target_fast(target, group_ids: torch.Tensor, ignore_label: Optional[int] = None):
    """Deep-supervision-aware wrapper with the trainer's _remap_target_for_group contract."""
    if isinstance(target, (list, tuple)):
        pieces = [remap_original_to_structured_fast(t, group_ids, ignore_label) for t in target]
        return [p[0] for p in pieces], [p[1] for p in pieces], pieces[0][2]
    return remap_original_to_structured_fast(target, group_ids, ignore_label)


def _gather_outputs(outputs, group_ids: torch.Tensor):
    if isinstance(outputs, (list, tuple)):
        return [gather_group_logits(o, group_ids) for o in outputs]
    return gather_group_logits(outputs, group_ids)


class GroupedSoftmaxFlatNet(nn.Module):
    """
    Unconditioned nnU-Net backbone with a 45-channel flat head, exposing the structured network's
    interface: forward(x, group_ids) returns each sample's 9-channel group logits (a list under deep
    supervision). all_groups=True returns the raw flat logits. encode/decode_from_skips let the
    trainer's reuse-encoder validation run one backbone pass for all queried groups.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    @property
    def decoder(self):
        return self.backbone.decoder

    def forward(
        self,
        x: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None,
        return_features: bool = False,
        all_groups: bool = False,
    ):
        if return_features:
            raise NotImplementedError("GroupedSoftmaxFlatNet does not expose decoder features")
        flat = self.backbone(x)
        if all_groups:
            return flat
        if group_ids is None:
            group_ids = torch.zeros((x.shape[0],), dtype=torch.long, device=x.device)
        return _gather_outputs(flat, group_ids)

    def encode(self, x: torch.Tensor):
        """Returns the full flat logits; decode_from_skips then gathers per query without recomputation."""
        return self.backbone(x)

    def decode_from_skips(self, flat, group_ids: Optional[torch.Tensor], return_features: bool = False):
        if return_features:
            raise NotImplementedError("GroupedSoftmaxFlatNet does not expose decoder features")
        first = flat[0] if isinstance(flat, (list, tuple)) else flat
        if group_ids is None:
            group_ids = torch.zeros((first.shape[0],), dtype=torch.long, device=first.device)
        return _gather_outputs(flat, group_ids)


Outputs = Union[torch.Tensor, List[torch.Tensor]]


def grouped_softmax_loss(
    flat_outputs: Outputs,
    target: Outputs,
    compute_loss: Callable[[Outputs, Outputs, Outputs, torch.Tensor], torch.Tensor],
    remap_target: Callable[[Outputs, torch.Tensor], Tuple[Outputs, Outputs, torch.Tensor]],
    num_groups: int = NUM_DYNAMIC_GROUPS,
    use_checkpoint: bool = True,
) -> torch.Tensor:
    """
    Mean over all groups of the structured loss on each group's gathered logits and remapped target.
    use_checkpoint recomputes each group's loss in backward so peak memory stays near one group's.
    """
    is_list = isinstance(flat_outputs, (list, tuple))
    tensors: Sequence[torch.Tensor] = list(flat_outputs) if is_list else [flat_outputs]
    batch = int(tensors[0].shape[0])
    device = tensors[0].device
    total = None
    for g in range(int(num_groups)):
        group_ids = torch.full((batch,), g, dtype=torch.long, device=device)
        remapped, valid, active = remap_target(target, group_ids)

        def group_loss(*outs, _ids=group_ids, _t=remapped, _v=valid, _a=active):
            gathered = [gather_group_logits(o, _ids) for o in outs]
            return compute_loss(gathered if is_list else gathered[0], _t, _v, _a)

        if use_checkpoint and torch.is_grad_enabled():
            loss_g = checkpoint(group_loss, *tensors, use_reentrant=False)
        else:
            loss_g = group_loss(*tensors)
        total = loss_g if total is None else total + loss_g
    return total / float(num_groups)
