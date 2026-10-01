"""Optional unchanged-raster fallback for insufficient registration coverage."""

from contextlib import contextmanager
import os
import warnings

import rasterio
from rasterio.shutil import copy as copy_raster


class InsufficientOverlapError(ValueError):
    """The registration region has insufficient usable source/reference coverage."""


@contextmanager
def handle_non_overlap(moving_image_path, fixed_image_path, output_image_path, *, enabled):
    """Handle only coverage failures, after raster handles have been closed."""
    try:
        yield
    except InsufficientOverlapError as exc:
        if not enabled:
            raise
        warnings.warn(
            f"Skipping coregistration of {moving_image_path} against {fixed_image_path}: {exc} "
            f"Writing the unchanged moving raster to {output_image_path}.",
            RuntimeWarning,
            stacklevel=3,
        )
        if os.path.exists(output_image_path) and os.path.samefile(moving_image_path, output_image_path):
            return
        os.makedirs(os.path.dirname(output_image_path) or ".", exist_ok=True)
        # GDAL's raster copy retains external masks and metadata as well as pixels.
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            copy_raster(moving_image_path, output_image_path, driver="GTiff")
