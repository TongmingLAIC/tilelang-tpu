# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Portable TPU online-softmax attention demo."""

from .flashattn import build_flashattn, run

__all__ = ["build_flashattn", "run"]
