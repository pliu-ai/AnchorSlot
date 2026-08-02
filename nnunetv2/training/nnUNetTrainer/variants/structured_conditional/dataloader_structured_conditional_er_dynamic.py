from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase

from .label_mapping_er_dynamic import (
    NUM_DYNAMIC_GROUPS,
    infer_present_groups_from_class_locations,
    sample_group_id_for_case,
)


class StructuredConditionalDataLoader3D(nnUNetDataLoaderBase):
    """
    3D dataloader that samples one dynamic group ID per case.

    Group sampling is biased towards present groups but can intentionally sample
    absent groups to provide negative conditional supervision.
    """

    def __init__(
        self,
        *args,
        p_present_group: float = 0.8,
        seed: Optional[int] = None,
        group_sampling_mode: str = "random",
        group_balance_power: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.p_present_group = float(np.clip(p_present_group, 0.0, 1.0))
        self.rng = np.random.default_rng(seed)
        self._case_present_group_cache: Dict[str, Sequence[int]] = {}
        self.group_sampling_mode = str(group_sampling_mode).strip().lower()
        if self.group_sampling_mode in {"", "none", "default"}:
            self.group_sampling_mode = "random"
        if self.group_sampling_mode not in {"random", "balanced"}:
            raise ValueError("group_sampling_mode must be 'random' or 'balanced'.")
        self.group_balance_power = float(group_balance_power)
        if self.group_balance_power < 0:
            raise ValueError("group_balance_power must be >= 0.")
        self._group_sample_counts = np.zeros((NUM_DYNAMIC_GROUPS,), dtype=np.int64)

    def set_epoch(self, epoch: int) -> None:
        del epoch
        self._group_sample_counts[:] = 0

    def _get_present_groups_for_case(self, case_key: str, properties: dict) -> Sequence[int]:
        if case_key in self._case_present_group_cache:
            return self._case_present_group_cache[case_key]

        present: Set[int] = infer_present_groups_from_class_locations(properties)
        present_sorted = tuple(sorted(present))
        self._case_present_group_cache[case_key] = present_sorted
        return present_sorted

    def _sample_balanced_group_id(self, present_group_ids: Sequence[int]) -> int:
        present_sorted = sorted({int(i) for i in present_group_ids if 0 <= int(i) < NUM_DYNAMIC_GROUPS})
        absent = [i for i in range(NUM_DYNAMIC_GROUPS) if i not in present_sorted]

        if len(present_sorted) == 0:
            candidates = list(range(NUM_DYNAMIC_GROUPS))
        elif bool(self.rng.random() < self.p_present_group):
            candidates = present_sorted
        elif len(absent) > 0:
            candidates = absent
        else:
            candidates = present_sorted

        counts = self._group_sample_counts[np.asarray(candidates, dtype=np.int64)].astype(np.float64)
        weights = np.power(counts + 1.0, -self.group_balance_power)
        weights = weights / np.clip(weights.sum(), a_min=1e-12, a_max=None)
        group_id = int(self.rng.choice(np.asarray(candidates, dtype=np.int64), p=weights))
        self._group_sample_counts[group_id] += 1
        return group_id

    def _sample_group_id(self, present_group_ids: Sequence[int]) -> int:
        if self.group_sampling_mode == "balanced":
            return self._sample_balanced_group_id(present_group_ids)
        return sample_group_id_for_case(
            present_group_ids=present_group_ids,
            p_present_group=self.p_present_group,
            rng=self.rng,
        )

    def generate_train_batch(self):
        selected_keys = self.get_indices()

        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        group_ids = np.zeros((self.batch_size,), dtype=np.int64)
        chosen_keys: List[str] = []

        for j, case_key in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(case_key)
            present_groups = self._get_present_groups_for_case(case_key, properties)
            group_ids[j] = self._sample_group_id(present_groups)

            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties["class_locations"])

            valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_bbox_ubs = np.minimum(shape, bbox_ubs)

            data_slice = tuple([slice(0, data.shape[0])] + [slice(i, k) for i, k in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg_slice = tuple([slice(0, seg.shape[0])] + [slice(i, k) for i, k in zip(valid_bbox_lbs, valid_bbox_ubs)])

            data = data[data_slice]
            seg = seg[seg_slice]

            padding = [(-min(0, bbox_lbs[d]), max(bbox_ubs[d] - shape[d], 0)) for d in range(dim)]
            padding = ((0, 0), *padding)
            data_all[j] = np.pad(data, padding, "constant", constant_values=0)
            seg_all[j] = np.pad(seg, padding, "constant", constant_values=-1)
            chosen_keys.append(case_key)

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all_t = torch.from_numpy(data_all).float()
                    seg_all_t = torch.from_numpy(seg_all).to(torch.int16)
                    group_ids_t = torch.from_numpy(group_ids).long()

                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        transformed = self.transforms(**{"image": data_all_t[b], "segmentation": seg_all_t[b]})
                        images.append(transformed["image"])
                        segs.append(transformed["segmentation"])

                    data_all_t = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all_t = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all_t = torch.stack(segs)

                    del images, segs

            return {
                "data": data_all_t,
                "target": seg_all_t,
                "keys": chosen_keys,
                "group_id": group_ids_t,
            }

        return {
            "data": data_all,
            "target": seg_all,
            "keys": chosen_keys,
            "group_id": group_ids,
        }
