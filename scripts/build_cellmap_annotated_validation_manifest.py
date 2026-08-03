#!/usr/bin/env python3
"""Keep only annotated crop/class pairs in a CellMap validation manifest."""

from nnunetv2.evaluation.cellmap_challenge import (
    build_annotated_manifest_entry_point,
)


if __name__ == "__main__":
    build_annotated_manifest_entry_point()
