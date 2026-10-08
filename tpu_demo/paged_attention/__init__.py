# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Portable TPU paged decode-attention demo."""

from .paged_attention import (build_paged_attention, index_tables, paged_rows, run,
                              torch_reference)

__all__ = [
    "build_paged_attention", "index_tables", "paged_rows", "run", "torch_reference"
]
