#!/usr/bin/env python3
"""CLI for the manifest-locked CellMap evaluation pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nnunetv2.evaluation.cellmap_pipeline import main


if __name__ == "__main__":
    main()
