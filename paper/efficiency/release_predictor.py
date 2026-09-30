"""StructuredConditionalPredictor from scripts/predict_structured_conditional.py of this repository."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "predict_structured_conditional.py"
_spec = importlib.util.spec_from_file_location("predict_structured_conditional", _SCRIPT)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
StructuredConditionalPredictor = _module.StructuredConditionalPredictor
