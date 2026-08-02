"""Structured conditional nnUNet trainer components for CellMap."""

from .trainer_structured_conditional import nnUNetTrainerStructuredConditional
from .trainer_structured_conditional_no_slot3 import nnUNetTrainerStructuredConditionalNoSlot3
from .trainer_structured_conditional_no_slot3_balanced_present import (
    nnUNetTrainerStructuredConditionalNoSlot3BalancedPresent,
)
from .trainer_structured_conditional_no_slot3_multi_condition import (
    nnUNetTrainerStructuredConditionalNoSlot3MultiCondition,
)
from .trainer_hierarchical_parallel_anchorslot import nnUNetTrainerHierarchicalParallelAnchorSlot
from .trainer_structured_conditional_no_slot3_er_in_cond_slot import (
    nnUNetTrainerStructuredConditionalNoSlot3ERInCondSlot,
)
from .trainer_structured_conditional_no_slot3_er_dynamic import (
    nnUNetTrainerStructuredConditionalNoSlot3ERDynamic,
)
from .trainer_structured_conditional_er_dynamic import (
    nnUNetTrainerStructuredConditionalERDynamic,
    nnUNetTrainerStructuredConditionalERDynamicClassBalancedCE,
    nnUNetTrainerStructuredConditionalERDynamicGroupBalanced,
    nnUNetTrainerStructuredConditionalERDynamicSlotBalancedCE,
)
from .trainer_structured_conditional_no_slot3_er_dynamic_slot_assignment import (
    nnUNetTrainerStructuredConditionalNoSlot3ERDynamicSlotAssignment,
)
from .trainer_structured_conditional_no_slot3_minimal_anchors import (
    nnUNetTrainerStructuredConditionalNoSlot3MinimalAnchors,
)
from .trainer_structured_conditional_no_slot3_mito_fixed import (
    nnUNetTrainerStructuredConditionalNoSlot3MitoFixed,
)
from .trainer_structured_conditional_no_slot3_mem_lum_consistency import (
    nnUNetTrainerMemLumConsistency,
    nnUNetTrainerStructuredConditionalNoSlot3MemLumConsistency,
)

try:
    from .trainer_structured_conditional_no_slot3_mednext import (
        nnUNetTrainerStructuredConditionalNoSlot3MedNeXt,
    )
except ImportError:
    nnUNetTrainerStructuredConditionalNoSlot3MedNeXt = None

try:
    from .trainer_mednext_32out_no_cond import nnUNetTrainerMedNeXt32OutNoCond
except ImportError:
    nnUNetTrainerMedNeXt32OutNoCond = None

try:
    from .trainer_structured_conditional_er_dynamic_mednext import (
        nnUNetTrainerStructuredConditionalERDynamicMedNeXt,
    )
except ImportError:
    nnUNetTrainerStructuredConditionalERDynamicMedNeXt = None

try:
    from .trainer_structured_conditional_asem import nnUNetTrainerStructuredConditionalASEM
except ImportError:
    nnUNetTrainerStructuredConditionalASEM = None

__all__ = [
    "nnUNetTrainerStructuredConditional",
    "nnUNetTrainerStructuredConditionalNoSlot3",
    "nnUNetTrainerStructuredConditionalNoSlot3BalancedPresent",
    "nnUNetTrainerStructuredConditionalNoSlot3MultiCondition",
    "nnUNetTrainerHierarchicalParallelAnchorSlot",
    "nnUNetTrainerStructuredConditionalNoSlot3ERInCondSlot",
    "nnUNetTrainerStructuredConditionalNoSlot3ERDynamic",
    "nnUNetTrainerStructuredConditionalERDynamic",
    "nnUNetTrainerStructuredConditionalERDynamicClassBalancedCE",
    "nnUNetTrainerStructuredConditionalERDynamicSlotBalancedCE",
    "nnUNetTrainerStructuredConditionalERDynamicGroupBalanced",
    "nnUNetTrainerStructuredConditionalNoSlot3ERDynamicSlotAssignment",
    "nnUNetTrainerStructuredConditionalNoSlot3MinimalAnchors",
    "nnUNetTrainerStructuredConditionalNoSlot3MitoFixed",
    "nnUNetTrainerMemLumConsistency",
    "nnUNetTrainerStructuredConditionalNoSlot3MemLumConsistency",
]

if nnUNetTrainerStructuredConditionalNoSlot3MedNeXt is not None:
    __all__.append("nnUNetTrainerStructuredConditionalNoSlot3MedNeXt")
if nnUNetTrainerMedNeXt32OutNoCond is not None:
    __all__.append("nnUNetTrainerMedNeXt32OutNoCond")
if nnUNetTrainerStructuredConditionalERDynamicMedNeXt is not None:
    __all__.append("nnUNetTrainerStructuredConditionalERDynamicMedNeXt")

try:
    from .trainer_structured_conditional_totalseg import nnUNetTrainerStructuredConditionalTotalSeg
except ImportError:
    nnUNetTrainerStructuredConditionalTotalSeg = None
