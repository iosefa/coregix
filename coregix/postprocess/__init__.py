"""Post-processing utilities for aligned rasters."""

from .edge_trim import EdgeTrimResult, trim_edge_invalid_pixels

__all__ = ["EdgeTrimResult", "trim_edge_invalid_pixels"]
