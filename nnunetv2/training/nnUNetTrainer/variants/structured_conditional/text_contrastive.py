from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class VisualTextContrastiveLoss(nn.Module):
    """
    Align sampled decoder voxel features with text prototypes indexed by original
    CellMap label IDs.

    The module is intentionally projection-free: projection heads live on the
    model so their parameters are optimized with the rest of the network.
    """

    def __init__(
        self,
        text_contrast_tau: float = 0.1,
        text_contrast_num_samples: int = 40,
        text_contrast_include_anchors: bool = True,
        text_contrast_include_active_dynamic: bool = True,
        contrast_all_present_labels: bool = False,
        ignore_label_ids: Optional[Iterable[int]] = None,
        anchor_label_ids: Sequence[int] = (1, 2, 3, 27),
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.tau = float(text_contrast_tau)
        self.num_samples = int(text_contrast_num_samples)
        self.include_anchors = bool(text_contrast_include_anchors)
        self.include_active_dynamic = bool(text_contrast_include_active_dynamic)
        self.contrast_all_present_labels = bool(contrast_all_present_labels)
        self.ignore_label_ids = {int(i) for i in (ignore_label_ids if ignore_label_ids is not None else (0, -1))}
        self.anchor_label_ids = tuple(int(i) for i in anchor_label_ids)
        self.eps = float(eps)
        self.last_stats: Dict[str, object] = {
            "num_samples": 0,
            "num_labels": 0,
            "sampled_label_ids": [],
            "missing_text_label_ids": [],
        }
        if self.tau <= 0:
            raise ValueError("text_contrast_tau must be > 0")
        if self.num_samples < 1:
            raise ValueError("text_contrast_num_samples must be >= 1")

    @staticmethod
    def _zero(*tensors: torch.Tensor) -> torch.Tensor:
        zero = None
        for tensor in tensors:
            if not torch.is_tensor(tensor) or tensor.numel() == 0:
                continue
            term = tensor.float().reshape(-1)[0] * 0.0
            zero = term if zero is None else zero + term
        if zero is None:
            raise ValueError("VisualTextContrastiveLoss._zero requires at least one non-empty tensor")
        return zero

    def _downsample_labels(self, original_label_map: torch.Tensor, spatial_size: Sequence[int]) -> torch.Tensor:
        labels = original_label_map
        if labels.ndim == len(spatial_size) + 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        if labels.ndim != len(spatial_size) + 1:
            raise ValueError(
                f"original_label_map must be [B, *spatial] or [B, 1, *spatial], got {tuple(original_label_map.shape)}"
            )
        labels_f = labels.float().unsqueeze(1)
        labels_ds = F.interpolate(labels_f, size=tuple(spatial_size), mode="nearest")
        return labels_ds[:, 0].long()

    def _valid_text_ids(self, valid_original_label_ids: Sequence[int] | torch.Tensor) -> set[int]:
        if torch.is_tensor(valid_original_label_ids):
            ids = valid_original_label_ids.detach().cpu().reshape(-1).tolist()
        else:
            ids = list(valid_original_label_ids)
        return {int(i) for i in ids if int(i) not in self.ignore_label_ids and int(i) > 0}

    def _selected_labels_for_item(
        self,
        labels_b: torch.Tensor,
        group_id: int,
        valid_text_ids: set[int],
        active_group_label_mapping: Mapping[int, Sequence[int]],
    ) -> Tuple[List[int], List[int]]:
        if self.contrast_all_present_labels:
            present = torch.unique(labels_b).detach().cpu().tolist()
            selected = sorted(
                int(i)
                for i in present
                if int(i) > 0 and int(i) not in self.ignore_label_ids and int(i) in valid_text_ids
            )
            missing = sorted(
                int(i)
                for i in present
                if int(i) > 0 and int(i) not in self.ignore_label_ids and int(i) not in valid_text_ids
            )
            return selected, missing

        requested: List[int] = []
        if self.include_anchors:
            requested.extend(self.anchor_label_ids)
        if self.include_active_dynamic:
            requested.extend(int(i) for i in active_group_label_mapping.get(int(group_id), ()))

        seen = set()
        selected = []
        missing = []
        for label_id in requested:
            label_id = int(label_id)
            if label_id in seen or label_id <= 0 or label_id in self.ignore_label_ids:
                continue
            seen.add(label_id)
            if label_id in valid_text_ids:
                selected.append(label_id)
            else:
                missing.append(label_id)
        return selected, missing

    def forward(
        self,
        feature_map: torch.Tensor,
        original_label_map: torch.Tensor,
        group_ids: torch.Tensor,
        text_embeddings_by_label: torch.Tensor,
        valid_original_label_ids: Sequence[int] | torch.Tensor,
        active_group_label_mapping: Mapping[int, Sequence[int]],
    ) -> torch.Tensor:
        self.last_stats = {
            "num_samples": 0,
            "num_labels": 0,
            "sampled_label_ids": [],
            "missing_text_label_ids": [],
        }
        if feature_map.ndim < 4:
            raise ValueError(f"feature_map must be [B, C, *spatial], got {tuple(feature_map.shape)}")
        if text_embeddings_by_label.ndim != 2:
            raise ValueError("text_embeddings_by_label must be [num_labels, text_dim]")

        labels_ds = self._downsample_labels(original_label_map, feature_map.shape[2:])
        group_ids = group_ids.to(device=feature_map.device).reshape(-1).long()
        if group_ids.numel() == 1 and labels_ds.shape[0] > 1:
            group_ids = group_ids.expand(labels_ds.shape[0])
        if group_ids.numel() != labels_ds.shape[0]:
            raise ValueError(f"group_ids batch mismatch: got {group_ids.numel()}, expected {labels_ds.shape[0]}")

        valid_text_ids = self._valid_text_ids(valid_original_label_ids)
        sampled_features: List[torch.Tensor] = []
        sampled_label_ids: List[torch.Tensor] = []
        sampled_label_set = set()
        missing_text_label_ids = set()

        features = feature_map.float()
        for b in range(labels_ds.shape[0]):
            selected, missing = self._selected_labels_for_item(
                labels_b=labels_ds[b],
                group_id=int(group_ids[b].item()),
                valid_text_ids=valid_text_ids,
                active_group_label_mapping=active_group_label_mapping,
            )
            missing_text_label_ids.update(missing)
            for label_id in selected:
                mask_flat = (labels_ds[b] == int(label_id)).reshape(-1)
                num_voxels = int(mask_flat.sum().item())
                if num_voxels <= 0:
                    continue
                flat_indices = torch.nonzero(mask_flat, as_tuple=False).reshape(-1)
                if flat_indices.numel() > self.num_samples:
                    perm = torch.randperm(flat_indices.numel(), device=flat_indices.device)[: self.num_samples]
                    flat_indices = flat_indices[perm]
                feat_flat = features[b].reshape(features.shape[1], -1).transpose(0, 1)
                sampled_features.append(feat_flat[flat_indices])
                sampled_label_ids.append(
                    torch.full((flat_indices.numel(),), int(label_id), dtype=torch.long, device=feature_map.device)
                )
                sampled_label_set.add(int(label_id))

        if not sampled_features:
            self.last_stats = {
                "num_samples": 0,
                "num_labels": 0,
                "sampled_label_ids": [],
                "missing_text_label_ids": sorted(missing_text_label_ids),
            }
            return self._zero(feature_map, text_embeddings_by_label)

        visual = torch.cat(sampled_features, dim=0).float()
        label_ids = torch.cat(sampled_label_ids, dim=0)
        prototype_label_ids = sorted(sampled_label_set)
        proto_index = torch.as_tensor(prototype_label_ids, dtype=torch.long, device=feature_map.device)
        text = text_embeddings_by_label.to(device=feature_map.device).float()[proto_index]

        visual = F.normalize(visual, dim=1, eps=self.eps)
        text = F.normalize(text, dim=1, eps=self.eps)
        logits = visual @ text.t()
        logits = logits / max(self.tau, self.eps)

        target_lookup = {label_id: i for i, label_id in enumerate(prototype_label_ids)}
        targets = torch.as_tensor(
            [target_lookup[int(label_id)] for label_id in label_ids.detach().cpu().tolist()],
            dtype=torch.long,
            device=feature_map.device,
        )
        loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            return self._zero(feature_map, text_embeddings_by_label)

        self.last_stats = {
            "num_samples": int(visual.shape[0]),
            "num_labels": int(len(prototype_label_ids)),
            "sampled_label_ids": prototype_label_ids,
            "missing_text_label_ids": sorted(missing_text_label_ids),
        }
        return loss
