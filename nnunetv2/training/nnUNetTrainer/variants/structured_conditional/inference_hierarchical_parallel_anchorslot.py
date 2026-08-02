from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import autocast

from nnunetv2.utilities.helpers import dummy_context

from .network_hierarchical_parallel_anchorslot import get_main_semantic_output


@torch.no_grad()
def predict_hierarchical_parallel(
    network: torch.nn.Module,
    image: torch.Tensor,
    use_amp: bool = True,
    return_hierarchy: bool = False,
):
    """Run all AnchorSlot groups in one encoder/decoder pass."""
    device = image.device
    amp_context = autocast(device.type, enabled=use_amp) if device.type == "cuda" else dummy_context()
    with amp_context:
        output = network(image, return_hierarchy=return_hierarchy)
    if return_hierarchy:
        result: Dict[str, torch.Tensor] = {
            "semantic_logits": get_main_semantic_output(output),
            "coarse_logits": output["coarse_logits"][0]
            if isinstance(output["coarse_logits"], (tuple, list))
            else output["coarse_logits"],
            "slot_logits": output["slot_logits"][0]
            if isinstance(output["slot_logits"], (tuple, list))
            else output["slot_logits"],
        }
        return result
    return get_main_semantic_output(output)


@torch.no_grad()
def predict_original_labels_hierarchical_parallel(
    network: torch.nn.Module,
    image: torch.Tensor,
    use_amp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = predict_hierarchical_parallel(network, image, use_amp=use_amp)
    return logits.argmax(dim=1, keepdim=True).long(), logits
