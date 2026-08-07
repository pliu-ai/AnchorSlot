from __future__ import annotations

from typing import Dict, List, Sequence, Union

import torch
import torch.nn.functional as F
from torch import nn

from .hierarchical_parallel_mapping import (
    FIXED_ORIGINAL_LABELS,
    GROUP_COARSE_START,
    NUM_ANCHOR_SLOTS,
    NUM_COARSE_CHANNELS,
    NUM_ORIGINAL_CLASSES,
    NUM_PARALLEL_GROUPS,
    ORIGINAL_TO_GROUP,
    ORIGINAL_TO_SLOT,
)


HierarchyOutput = Dict[str, Union[torch.Tensor, List[torch.Tensor]]]


class ParallelAnchorSlotHead(nn.Module):
    """Generate all group-specific slot classifiers from shared anchor codes."""

    def __init__(self, in_channels: int, code_dim: int = 64) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.code_dim = int(code_dim)
        self.weight_projection = nn.Linear(self.code_dim, self.in_channels)
        self.bias_projection = nn.Linear(self.code_dim, 1)

        nn.init.normal_(self.weight_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.weight_projection.bias)
        nn.init.zeros_(self.bias_projection.weight)
        nn.init.zeros_(self.bias_projection.bias)

    def forward(self, features: torch.Tensor, pair_codes: torch.Tensor) -> torch.Tensor:
        if features.shape[1] != self.in_channels:
            raise ValueError(
                f"feature width mismatch: got {features.shape[1]}, expected {self.in_channels}"
            )
        if pair_codes.ndim not in (3, 4):
            raise ValueError("pair_codes must have shape [G, S, C] or [B, G, S, C]")
        if pair_codes.ndim == 4 and pair_codes.shape[0] != features.shape[0]:
            raise ValueError("batch-conditioned pair codes must match the feature batch")
        weights = F.normalize(self.weight_projection(pair_codes), dim=-1)
        features = F.normalize(features, dim=1)
        biases = self.bias_projection(pair_codes).squeeze(-1)
        if pair_codes.ndim == 3:
            logits = torch.einsum("bc...,gsc->bgs...", features, weights)
            bias_shape = (1, *biases.shape, *([1] * (features.ndim - 2)))
        else:
            logits = torch.einsum("bc...,bgsc->bgs...", features, weights)
            bias_shape = (*biases.shape, *([1] * (features.ndim - 2)))
        return logits + biases.view(bias_shape)


class HierarchicalParallelAnchorSlotUNet(nn.Module):
    """
    Single-pass hierarchical AnchorSlot network.

    The decoder predicts a coarse taxonomy in parallel with two fine anchor
    slots per dynamic group. Coarse and within-group log-probabilities are
    composed into a normalized 32-class semantic distribution.
    """

    def __init__(self, backbone: nn.Module, code_dim: int = 64) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder = backbone.encoder
        self.decoder = backbone.decoder
        self.num_output_channels = NUM_ORIGINAL_CLASSES
        self.num_groups = NUM_PARALLEL_GROUPS
        self.num_slots = NUM_ANCHOR_SLOTS
        self.code_dim = int(code_dim)

        if any(int(layer.out_channels) != NUM_COARSE_CHANNELS for layer in self.decoder.seg_layers):
            raise ValueError(
                f"backbone segmentation heads must have {NUM_COARSE_CHANNELS} channels"
            )

        decoder_stage_channels = [int(layer.in_channels) for layer in self.decoder.seg_layers]
        self.group_embeddings = nn.Parameter(torch.empty(self.num_groups, self.code_dim))
        self.anchor_slot_embeddings = nn.Parameter(torch.empty(self.num_slots, self.code_dim))
        self.pair_encoder = nn.Sequential(
            nn.LayerNorm(self.code_dim),
            nn.Linear(self.code_dim, self.code_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.code_dim, self.code_dim),
        )
        self.slot_heads = nn.ModuleList(
            [ParallelAnchorSlotHead(channels, code_dim=self.code_dim) for channels in decoder_stage_channels]
        )
        nn.init.normal_(self.group_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.anchor_slot_embeddings, mean=0.0, std=0.02)

        fixed_coarse_channels = list(range(1, 1 + len(FIXED_ORIGINAL_LABELS)))
        self.register_buffer(
            "fixed_original_labels",
            torch.as_tensor(FIXED_ORIGINAL_LABELS, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "fixed_coarse_channels",
            torch.as_tensor(fixed_coarse_channels, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "dynamic_original_labels",
            torch.as_tensor(
                [label for label in range(NUM_ORIGINAL_CLASSES) if ORIGINAL_TO_GROUP[label] >= 0],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "dynamic_group_ids",
            torch.as_tensor(
                [ORIGINAL_TO_GROUP[label] for label in range(NUM_ORIGINAL_CLASSES) if ORIGINAL_TO_GROUP[label] >= 0],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "dynamic_slot_ids",
            torch.as_tensor(
                [ORIGINAL_TO_SLOT[label] for label in range(NUM_ORIGINAL_CLASSES) if ORIGINAL_TO_GROUP[label] >= 0],
                dtype=torch.long,
            ),
            persistent=False,
        )

    def _pair_codes(self) -> torch.Tensor:
        pair_codes = self.group_embeddings[:, None, :] + self.anchor_slot_embeddings[None, :, :]
        return self.pair_encoder(pair_codes)

    def compose_semantic_log_probs(
        self,
        coarse_logits: torch.Tensor,
        slot_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compose the hierarchy into normalized log-probabilities over 32 labels."""
        if coarse_logits.shape[1] != NUM_COARSE_CHANNELS:
            raise ValueError("coarse logits have an invalid channel count")
        if slot_logits.shape[1:3] != (self.num_groups, self.num_slots):
            raise ValueError("slot logits must have shape [B, num_groups, num_slots, ...]")

        coarse_prob_dtype = (
            torch.float32 if coarse_logits.dtype in (torch.float16, torch.bfloat16) else coarse_logits.dtype
        )
        slot_prob_dtype = (
            torch.float32 if slot_logits.dtype in (torch.float16, torch.bfloat16) else slot_logits.dtype
        )
        coarse_log_probs = F.log_softmax(coarse_logits, dim=1, dtype=coarse_prob_dtype)
        slot_log_probs = F.log_softmax(slot_logits, dim=2, dtype=slot_prob_dtype)
        # CUDA autocast may evaluate log_softmax in fp32 even when its input is
        # fp16. Allocate from the promoted log-probability dtype so indexed
        # assignments never mix fp16 destinations with fp32 sources.
        composition_dtype = torch.promote_types(coarse_log_probs.dtype, slot_log_probs.dtype)
        coarse_log_probs = coarse_log_probs.to(dtype=composition_dtype)
        slot_log_probs = slot_log_probs.to(dtype=composition_dtype)
        semantic = coarse_log_probs.new_empty(
            (coarse_logits.shape[0], NUM_ORIGINAL_CLASSES, *coarse_logits.shape[2:])
        )
        semantic[:, 0] = coarse_log_probs[:, 0]
        semantic[:, self.fixed_original_labels] = coarse_log_probs[:, self.fixed_coarse_channels]

        dynamic_coarse = coarse_log_probs[:, GROUP_COARSE_START + self.dynamic_group_ids]
        dynamic_slots = slot_log_probs[:, self.dynamic_group_ids, self.dynamic_slot_ids]
        semantic[:, self.dynamic_original_labels] = dynamic_coarse + dynamic_slots
        return semantic

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        skips: List[torch.Tensor] = []
        for stage in self.encoder.stages:
            x = stage(x)
            skips.append(x)
        return skips

    def decode_from_skips(self, skips: List[torch.Tensor], return_hierarchy: bool = False):
        lres_input = skips[-1]
        semantic_outputs: List[torch.Tensor] = []
        coarse_outputs: List[torch.Tensor] = []
        slot_outputs: List[torch.Tensor] = []
        pair_codes = self._pair_codes()

        for stage_idx in range(len(self.decoder.stages)):
            x = self.decoder.transpconvs[stage_idx](lres_input)
            x = torch.cat((x, skips[-(stage_idx + 2)]), dim=1)
            x = self.decoder.stages[stage_idx](x)

            emit = self.decoder.deep_supervision or stage_idx == len(self.decoder.stages) - 1
            if emit:
                coarse_logits = self.decoder.seg_layers[stage_idx](x)
                slot_logits = self.slot_heads[stage_idx](x, pair_codes)
                semantic_logits = self.compose_semantic_log_probs(coarse_logits, slot_logits)
                semantic_outputs.append(semantic_logits)
                coarse_outputs.append(coarse_logits)
                slot_outputs.append(slot_logits)
            lres_input = x

        semantic_outputs.reverse()
        coarse_outputs.reverse()
        slot_outputs.reverse()
        if self.decoder.deep_supervision:
            if return_hierarchy:
                return {
                    "semantic_logits": semantic_outputs,
                    "coarse_logits": coarse_outputs,
                    "slot_logits": slot_outputs,
                }
            return semantic_outputs

        if return_hierarchy:
            return {
                "semantic_logits": semantic_outputs[0],
                "coarse_logits": coarse_outputs[0],
                "slot_logits": slot_outputs[0],
            }
        return semantic_outputs[0]

    def forward(self, x: torch.Tensor, return_hierarchy: bool = False):
        return self.decode_from_skips(self.encode(x), return_hierarchy=return_hierarchy)


def get_main_semantic_output(output: Union[torch.Tensor, Sequence[torch.Tensor], HierarchyOutput]) -> torch.Tensor:
    if isinstance(output, dict):
        output = output["semantic_logits"]
    if isinstance(output, (tuple, list)):
        return output[0]
    return output
