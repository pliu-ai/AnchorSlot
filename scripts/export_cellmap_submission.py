#!/usr/bin/env python3
"""Export AnchorSlot hard-label predictions to a CellMap submission Zarr."""

from nnunetv2.evaluation.cellmap_challenge import export_entry_point


if __name__ == "__main__":
    export_entry_point()
