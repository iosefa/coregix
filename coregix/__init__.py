"""Joint raster coregistration for geospatial imagery."""

from .pipelines.alignment import align_image_pair

__all__ = ["align_image_pair"]
