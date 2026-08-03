#!/usr/bin/env python3
"""Validate a CellMap submission Zarr against the exact manifest used to export it."""

from nnunetv2.evaluation.cellmap_challenge import validate_entry_point


if __name__ == "__main__":
    validate_entry_point()
