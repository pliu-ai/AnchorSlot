from __future__ import annotations

import warnings
import re
from typing import Optional, Sequence, Tuple

import torch
from torch import nn


class GroupConditionEncoder(nn.Module):
    """
    Converts an integer dynamic-group ID into a learnable condition vector.

    The output vector is used by FiLM modules in the decoder/head side.
    """

    def __init__(self, num_groups: int, embedding_dim: int = 64, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        self.num_groups = int(num_groups)
        self.embedding_dim = int(embedding_dim)
        hidden = int(hidden_dim) if hidden_dim is not None else int(embedding_dim)

        if self.num_groups <= 0:
            raise ValueError("num_groups must be > 0")

        self.embedding = nn.Embedding(self.num_groups, self.embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
        )
        self.output_dim = hidden

    def _normalize_group_ids(
        self,
        group_ids: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not torch.is_tensor(group_ids):
            group_ids = torch.as_tensor(group_ids, device=device)
        group_ids = group_ids.to(device=device).reshape(-1).long()

        if group_ids.numel() == 1 and batch_size > 1:
            group_ids = group_ids.expand(batch_size)
        if group_ids.numel() != batch_size:
            raise ValueError(f"group_ids batch mismatch: got {group_ids.numel()}, expected {batch_size}")
        return group_ids.clamp(min=0, max=self.num_groups - 1)

    def forward(self, group_ids: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        group_ids = self._normalize_group_ids(group_ids, batch_size=batch_size, device=device)
        emb = self.embedding(group_ids)
        return self.mlp(emb)


class TextConditionEncoder(nn.Module):
    """
    Maps a dynamic-group ID to a condition vector via FROZEN text-prototype
    embeddings (e.g. centered BioMedCLIP organelle prototypes) projected into the
    FiLM condition space.

    Drop-in replacement for GroupConditionEncoder: identical forward signature and
    an `output_dim` attribute, so the decoder FiLM modules are unchanged. The text
    matrix is a frozen buffer; only the projection MLP is learnable (matching TAK,
    which freezes its CLIP text anchors and learns the mapping into feature space).
    """

    def __init__(
        self,
        num_groups: int,
        group_text_matrix: torch.Tensor,
        hidden_dim: int = 64,
        learnable_text: bool = False,
    ) -> None:
        super().__init__()
        self.num_groups = int(num_groups)
        if self.num_groups <= 0:
            raise ValueError("num_groups must be > 0")
        gt = group_text_matrix.detach().float()
        if gt.ndim != 2 or gt.shape[0] != self.num_groups:
            raise ValueError(
                f"group_text_matrix must be [num_groups={self.num_groups}, D], got {tuple(gt.shape)}"
            )
        self.text_dim = int(gt.shape[1])
        self.learnable_text = bool(learnable_text)
        if self.learnable_text:
            # text-INITIALIZED but adaptable: keeps the semantic prior yet lets the
            # prototypes specialize (removes the frozen-grounding ceiling).
            self.group_text = nn.Parameter(gt.clone())
        else:
            # frozen prototypes (persistent so they travel with the checkpoint).
            self.register_buffer("group_text", gt, persistent=True)

        hidden = int(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.text_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
        )
        self.output_dim = hidden

    def _normalize_group_ids(
        self,
        group_ids: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not torch.is_tensor(group_ids):
            group_ids = torch.as_tensor(group_ids, device=device)
        group_ids = group_ids.to(device=device).reshape(-1).long()
        if group_ids.numel() == 1 and batch_size > 1:
            group_ids = group_ids.expand(batch_size)
        if group_ids.numel() != batch_size:
            raise ValueError(f"group_ids batch mismatch: got {group_ids.numel()}, expected {batch_size}")
        return group_ids.clamp(min=0, max=self.num_groups - 1)

    def forward(self, group_ids: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        group_ids = self._normalize_group_ids(group_ids, batch_size=batch_size, device=device)
        emb = self.group_text[group_ids]  # (B, text_dim), frozen
        return self.mlp(emb)


class TextConditionedGroupEmbedding(nn.Module):
    """
    Learned group condition encoder augmented with precomputed BioMedCLIP group
    text embeddings.

    This module has the same forward signature and `output_dim` contract as
    GroupConditionEncoder. When use_text_conditioning=False it delegates to the
    learned encoder unchanged. Training never imports or constructs BioMedCLIP;
    it only consumes a precomputed group_text_matrix.
    """

    def __init__(
        self,
        num_groups: int,
        cond_dim: int = 64,
        group_text_matrix: Optional[torch.Tensor] = None,
        use_text_conditioning: bool = False,
        text_fusion: str = "concat_mlp",
        freeze_text_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.num_groups = int(num_groups)
        self.cond_dim = int(cond_dim)
        self.use_text_conditioning = bool(use_text_conditioning)
        self.text_fusion = str(text_fusion).strip().lower()
        self.freeze_text_embeddings = bool(freeze_text_embeddings)
        self.learned = GroupConditionEncoder(num_groups=self.num_groups, embedding_dim=self.cond_dim)
        self.output_dim = self.learned.output_dim

        if not self.use_text_conditioning:
            return
        if group_text_matrix is None:
            raise ValueError("use_text_conditioning=True requires group_text_matrix [num_groups, D]")
        if self.text_fusion not in {"concat_mlp", "add", "text_only"}:
            raise ValueError("text_fusion must be one of: concat_mlp, add, text_only")

        gt = group_text_matrix.detach().float()
        if gt.ndim != 2 or gt.shape[0] != self.num_groups:
            raise ValueError(
                f"group_text_matrix must be [num_groups={self.num_groups}, D], got {tuple(gt.shape)}"
            )
        self.text_dim = int(gt.shape[1])
        if self.freeze_text_embeddings:
            self.register_buffer("group_text", gt, persistent=True)
        else:
            self.group_text = nn.Parameter(gt.clone())

        self.text_projection = nn.Sequential(
            nn.Linear(self.text_dim, self.output_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.output_dim, self.output_dim),
        )
        if self.text_fusion == "concat_mlp":
            self.fusion_mlp = nn.Sequential(
                nn.Linear(self.output_dim * 2, self.output_dim),
                nn.SiLU(inplace=True),
                nn.Linear(self.output_dim, self.output_dim),
            )

    def forward(self, group_ids: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        learned_e = self.learned(group_ids, batch_size=batch_size, device=device)
        if not self.use_text_conditioning:
            return learned_e

        group_ids = self.learned._normalize_group_ids(group_ids, batch_size=batch_size, device=device)
        text_e = self.group_text[group_ids].to(device=device, dtype=learned_e.dtype)
        text_e = self.text_projection(text_e)
        if self.text_fusion == "concat_mlp":
            return self.fusion_mlp(torch.cat([learned_e, text_e], dim=1))
        if self.text_fusion == "add":
            return learned_e + text_e
        if self.text_fusion == "text_only":
            return text_e
        raise RuntimeError(f"unexpected text_fusion={self.text_fusion}")


def build_group_text_matrix(
    text_emb_path: str,
    group_label_ids,
    key: str = "per_label_mean_centered",
) -> torch.Tensor:
    """
    Build a [num_groups, D] text-prototype matrix: each group's prototype is the
    L2-normalized mean of its member original-label text embeddings.

    Args:
        text_emb_path: path to the saved text-embedding dict (see
            build_organelle_text_embeddings.py). Must contain `key` as a
            [num_original_labels(+bg), D] tensor indexed by ORIGINAL label id.
        group_label_ids: ordered iterable (one entry per group, in group_id order)
            of tuples of original label ids belonging to that group.
    """
    blob = torch.load(text_emb_path, map_location="cpu", weights_only=False)
    if key not in blob:
        raise KeyError(f"text embedding file {text_emb_path} has no key '{key}'; keys={list(blob)}")
    per_label = blob[key].float()  # (L, D) indexed by original label id
    rows = []
    for labels in group_label_ids:
        idx = [int(l) for l in labels]
        proto = per_label[idx].mean(dim=0)
        rows.append(torch.nn.functional.normalize(proto, dim=-1))
    return torch.stack(rows, dim=0)  # (num_groups, D)


def _group_lookup_keys(group_key: str, display_name: str, group_index: int) -> Tuple[str, ...]:
    display_spaced = re.sub(r"(?<!^)(?=[A-Z])", "_", str(display_name)).replace(" ", "_")
    display_snake = display_spaced.replace("__", "_")
    display_lower = display_snake.lower()
    display_compact = display_lower.replace("_", "")
    er_alias = "er" if display_lower == "e_r_in_cond_slot" else display_lower
    return (
        str(group_key),
        str(group_key).lower(),
        f"{group_key}_{display_lower}",
        f"{group_key.lower()}_{display_lower}",
        f"{group_key}_{display_compact}",
        f"{group_key.lower()}_{display_compact}",
        f"{group_key}_{er_alias}",
        f"{group_key.lower()}_{er_alias}",
        display_snake,
        display_lower,
        display_compact,
        er_alias,
        str(group_index),
        str(group_index + 1),
    )


def build_group_text_matrix_from_group_embeddings(
    text_emb_path: str,
    group_keys: Sequence[str],
    group_display_names: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    """
    Build [num_groups, D] from the new BioMedCLIP text prior artifact.

    The artifact is expected to contain:
        group_embeddings: dict[group_id_or_name] -> Tensor[D]

    Missing groups are warned and filled with zeros so a partially edited JSON can
    still be inspected. A file with no usable group embedding raises a clear error.
    """
    blob = torch.load(text_emb_path, map_location="cpu", weights_only=False)
    if "group_embeddings" not in blob or not isinstance(blob["group_embeddings"], dict):
        raise KeyError(f"text embedding file {text_emb_path} must contain dict key 'group_embeddings'")

    group_embeddings = blob["group_embeddings"]
    normalized_group_embeddings = {str(k).lower(): v for k, v in group_embeddings.items()}
    display_names = group_display_names if group_display_names is not None else group_keys
    first = None
    for value in group_embeddings.values():
        if torch.is_tensor(value):
            first = value.detach().float().reshape(-1)
            break
    if first is None:
        raise ValueError(f"text embedding file {text_emb_path} has no tensor-valued group embeddings")

    rows = []
    missing = []
    for idx, key in enumerate(group_keys):
        display = display_names[idx] if idx < len(display_names) else key
        value = None
        for lookup_key in _group_lookup_keys(str(key), str(display), idx):
            if lookup_key in group_embeddings:
                value = group_embeddings[lookup_key]
                break
            if lookup_key.lower() in normalized_group_embeddings:
                value = normalized_group_embeddings[lookup_key.lower()]
                break
        if value is None:
            missing.append(str(key))
            rows.append(torch.zeros_like(first))
            continue
        row = value.detach().float().reshape(-1)
        if row.shape != first.shape:
            raise ValueError(
                f"group embedding '{key}' has shape {tuple(row.shape)}, expected {tuple(first.shape)}"
            )
        rows.append(torch.nn.functional.normalize(row, dim=-1))
    if missing:
        warnings.warn(
            f"text embedding file {text_emb_path} is missing group embeddings for: {', '.join(missing)}; using zeros",
            RuntimeWarning,
        )
    return torch.stack(rows, dim=0)


class FiLMModulation(nn.Module):
    """
    Feature-wise linear modulation (FiLM): y = gamma * x + beta.

    The affine layer is initialized as identity so conditional modulation starts
    from a stable no-op behavior.
    """

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.affine = nn.Linear(int(cond_dim), 2 * self.channels)

        # Identity initialization: gamma = 1, beta = 0.
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)
        with torch.no_grad():
            self.affine.bias[: self.channels].fill_(1.0)

    def forward(self, x: torch.Tensor, cond_vector: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.affine(cond_vector)
        gamma, beta = torch.split(gamma_beta, self.channels, dim=1)
        view_shape = [x.shape[0], self.channels] + [1] * (x.ndim - 2)
        gamma = gamma.view(*view_shape)
        beta = beta.view(*view_shape)
        return gamma * x + beta
