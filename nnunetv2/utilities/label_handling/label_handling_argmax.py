"""Label manager for mutually exclusive labels trained as independent sigmoid regions.

nnU-Net's region export assigns labels in regions_class_order wherever a region probability exceeds 0.5, so later
regions overwrite earlier ones. For one-region-per-label training of mutually exclusive labels, this manager instead
assigns each voxel the label with the highest probability, and background where no probability exceeds 0.5.
Selected through plans['label_manager'] = 'ArgmaxRegionLabelManager'.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

from nnunetv2.utilities.label_handling.label_handling import LabelManager


class ArgmaxRegionLabelManager(LabelManager):
    THRESHOLD = 0.5

    def __init__(self, label_dict: dict, regions_class_order, force_use_labels: bool = False, inference_nonlin=None):
        super().__init__(label_dict, regions_class_order, force_use_labels, inference_nonlin)
        # nnU-Net only treats a label-dict entry as a region if it lists more than one label. Here every foreground
        # label is its own single-label region, so region mode (sigmoid outputs) is forced explicitly.
        if not force_use_labels and regions_class_order is not None and not self._has_regions:
            self._has_regions = True
            self._regions = self._get_regions()
            if inference_nonlin is None:
                self.inference_nonlin = torch.sigmoid

    def convert_probabilities_to_segmentation(self, predicted_probabilities: Union[np.ndarray, torch.Tensor]) -> \
            Union[np.ndarray, torch.Tensor]:
        if not self.has_regions:
            return super().convert_probabilities_to_segmentation(predicted_probabilities)
        if predicted_probabilities.shape[0] != self.num_segmentation_heads:
            raise ValueError(f"expected {self.num_segmentation_heads} channels, got {predicted_probabilities.shape[0]}")
        order = list(self.regions_class_order)
        if isinstance(predicted_probabilities, np.ndarray):
            best = predicted_probabilities.argmax(0)
            segmentation = np.asarray(order, dtype=np.uint16)[best]
            segmentation[predicted_probabilities.max(0) <= self.THRESHOLD] = 0
            return segmentation
        lut = torch.as_tensor(order, dtype=torch.int16, device=predicted_probabilities.device)
        best_prob, best = predicted_probabilities.max(0)
        segmentation = lut[best]
        segmentation[best_prob <= self.THRESHOLD] = 0
        return segmentation
