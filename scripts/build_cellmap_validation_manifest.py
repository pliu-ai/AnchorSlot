#!/usr/bin/env python3
"""Build an all-48-class manifest from legacy validation predictions."""

from nnunetv2.evaluation.cellmap_challenge import build_manifest_entry_point


if __name__ == "__main__":
    build_manifest_entry_point()
