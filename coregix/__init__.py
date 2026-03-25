"""Standalone elastix alignment package."""

from .pipelines.alignment import AlignmentResult, align_image_pair

__all__ = ["AlignmentResult", "align_image_pair"]
