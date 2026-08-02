from __future__ import annotations

import importlib
import os
from types import ModuleType


def get_label_mapping() -> ModuleType:
    module_name = os.environ.get(
        "NNUNET_STRUCTCOND_SYNAPSE_MAPPING_MODULE",
        "nnunetv2.training.nnUNetTrainer.variants.structured_conditional.label_mapping_synapse_no_slot3",
    )
    return importlib.import_module(module_name)
