import numpy as np
import torch


def merge_candidates(best_probability, best_label, probabilities, labels, complement):
    """Merge task probabilities after tile/TTA aggregation and grid resampling."""
    for channel, label in enumerate(labels):
        p = probabilities[channel]
        eligible = p > complement
        win = eligible & ((p > best_probability) | ((p == best_probability) & (label < best_label)))
        best_probability[win] = p[win]
        best_label[win] = label


def restore_grid(segmentation, properties, plans_manager):
    restored = np.zeros(properties['shape_before_cropping'], dtype=np.uint8)
    slices = tuple(slice(a, b) for a, b in properties['bbox_used_for_cropping'])
    restored[slices] = segmentation
    return restored.transpose(plans_manager.transpose_backward)


def resample_logits(logits, properties, plans_manager, configuration_manager):
    spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]
    z = configuration_manager.resampling_fn_probabilities(
        logits.cpu(), properties['shape_after_cropping_and_before_resampling'],
        configuration_manager.spacing, spacing)
    return torch.as_tensor(z).float()
