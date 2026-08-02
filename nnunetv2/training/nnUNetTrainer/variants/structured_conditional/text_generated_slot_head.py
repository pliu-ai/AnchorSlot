from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class TextGeneratedSlotHead(nn.Module):
    """
    Generate dynamic AnchorSlot logits from label text embeddings.

    The fixed structured head still predicts all output channels. This module
    only produces the K group-dependent dynamic slot logits that can replace or
    residual-add to the fixed head's dynamic channels.
    """

    def __init__(
        self,
        in_channels: int,
        text_embedding_dim: int,
        num_slots: int = 3,
        hidden_dim: int = 512,
        use_bias: bool = True,
        normalize_weight: bool = True,
        inactive_logit: float = -1e4,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.text_embedding_dim = int(text_embedding_dim)
        self.num_slots = int(num_slots)
        self.use_bias = bool(use_bias)
        self.normalize_weight = bool(normalize_weight)
        self.inactive_logit = float(inactive_logit)
        hidden_dim = int(hidden_dim)

        if self.in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        if self.text_embedding_dim <= 0:
            raise ValueError("text_embedding_dim must be > 0")
        if self.num_slots <= 0:
            raise ValueError("num_slots must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")

        self.weight_mlp = nn.Sequential(
            nn.Linear(self.text_embedding_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, self.in_channels),
        )
        if self.use_bias:
            self.bias_mlp = nn.Sequential(
                nn.Linear(self.text_embedding_dim, hidden_dim),
                nn.SiLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.bias_mlp = None

        self.last_stats: Dict[str, object] = {
            "active_slots": 0,
            "inactive_slots": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    def _resolve_text_embeddings(
        self,
        group_ids: torch.Tensor,
        slot_label_text_embeddings: torch.Tensor | Mapping[str, torch.Tensor],
        group_to_slot_labels: Mapping[int, Sequence[str]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = int(group_ids.numel())
        active_mask = torch.zeros((b, self.num_slots), dtype=torch.bool, device=device)

        if torch.is_tensor(slot_label_text_embeddings):
            embeddings = slot_label_text_embeddings.to(device=device)
            if embeddings.ndim != 3:
                raise ValueError(
                    "slot_label_text_embeddings tensor must be [num_groups, num_slots, text_dim], "
                    f"got {tuple(embeddings.shape)}"
                )
            if embeddings.shape[1] != self.num_slots or embeddings.shape[2] != self.text_embedding_dim:
                raise ValueError(
                    "slot_label_text_embeddings tensor shape mismatch: "
                    f"expected [G, {self.num_slots}, {self.text_embedding_dim}], got {tuple(embeddings.shape)}"
                )
            if int(group_ids.max().item()) >= embeddings.shape[0] or int(group_ids.min().item()) < 0:
                raise ValueError("group_ids are outside slot_label_text_embeddings group dimension")
            text_batch = embeddings[group_ids]
            for i, group_id in enumerate(group_ids.detach().cpu().tolist()):
                active_mask[i, : len(tuple(group_to_slot_labels.get(int(group_id), ())))] = True
            return text_batch, active_mask

        text_batch = torch.zeros(
            (b, self.num_slots, self.text_embedding_dim),
            dtype=torch.float32,
            device=device,
        )
        for i, group_id in enumerate(group_ids.detach().cpu().tolist()):
            labels = tuple(group_to_slot_labels.get(int(group_id), ()))
            if len(labels) > self.num_slots:
                raise ValueError(f"group_id={group_id} has {len(labels)} labels but num_slots={self.num_slots}")
            for slot_idx, label_name in enumerate(labels):
                if label_name not in slot_label_text_embeddings:
                    raise KeyError(f"Missing text embedding for dynamic slot label {label_name!r}")
                value = slot_label_text_embeddings[label_name]
                text_batch[i, slot_idx] = value.to(device=device, dtype=text_batch.dtype).reshape(-1)
                active_mask[i, slot_idx] = True
        return text_batch, active_mask

    def forward(
        self,
        feature_map: torch.Tensor,
        group_ids: torch.Tensor,
        slot_label_text_embeddings: torch.Tensor | Mapping[str, torch.Tensor],
        group_to_slot_labels: Mapping[int, Sequence[str]],
    ) -> torch.Tensor:
        if feature_map.ndim != 5:
            raise ValueError(f"feature_map must be [B, C, D, H, W], got {tuple(feature_map.shape)}")
        if int(feature_map.shape[1]) != self.in_channels:
            raise ValueError(f"feature_map has {feature_map.shape[1]} channels, expected {self.in_channels}")

        group_ids = group_ids.reshape(-1).to(device=feature_map.device, dtype=torch.long)
        if group_ids.numel() == 1 and feature_map.shape[0] > 1:
            group_ids = group_ids.expand(feature_map.shape[0])
        if group_ids.numel() != feature_map.shape[0]:
            raise ValueError(f"group_ids batch mismatch: got {group_ids.numel()}, expected {feature_map.shape[0]}")

        text_batch, active_mask = self._resolve_text_embeddings(
            group_ids=group_ids,
            slot_label_text_embeddings=slot_label_text_embeddings,
            group_to_slot_labels=group_to_slot_labels,
            device=feature_map.device,
        )
        try:
            param_dtype = next(self.parameters()).dtype
        except StopIteration:
            param_dtype = feature_map.dtype
        text_batch = text_batch.to(dtype=param_dtype)
        weights = self.weight_mlp(text_batch)
        if self.normalize_weight:
            weights = F.normalize(weights.float(), dim=-1)
        weights = weights.to(dtype=feature_map.dtype)
        logits = torch.einsum("bcdhw,bkc->bkdhw", feature_map, weights)
        if self.bias_mlp is not None:
            bias = self.bias_mlp(text_batch).to(dtype=logits.dtype).squeeze(-1)
            logits = logits + bias[:, :, None, None, None]

        logits = logits.masked_fill(~active_mask[:, :, None, None, None], self.inactive_logit)
        active_values = logits[active_mask[:, :, None, None, None].expand_as(logits)]
        if active_values.numel() > 0:
            stats_values = active_values.detach().float()
            self.last_stats = {
                "active_slots": int(active_mask.sum().item()),
                "inactive_slots": int((~active_mask).sum().item()),
                "mean": float(stats_values.mean().item()),
                "std": float(stats_values.std(unbiased=False).item()),
                "min": float(stats_values.min().item()),
                "max": float(stats_values.max().item()),
            }
        else:
            self.last_stats = {
                "active_slots": 0,
                "inactive_slots": int((~active_mask).sum().item()),
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        return logits
