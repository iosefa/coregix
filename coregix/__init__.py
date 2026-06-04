"""Pairwise raster coregistration for geospatial imagery."""

from .evaluation import (
    FeatureAlignmentError,
    VectorAlignmentResult,
    evaluate_vector_alignment,
)
from .pipelines.alignment import AlignmentResult, align_image_pair
from .pipelines.apply_transform import ApplyTransformResult, apply_coregix_transform

__all__ = [
    "AlignmentResult",
    "ApplyTransformResult",
    "FeatureAlignmentError",
    "VectorAlignmentResult",
    "align_image_pair",
    "apply_coregix_transform",
    "evaluate_vector_alignment",
]
