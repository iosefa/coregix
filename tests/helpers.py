from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.transform import Affine, from_origin


DEFAULT_CRS = "EPSG:32610"
DEFAULT_TRANSFORM = from_origin(100.0, 200.0, 1.0, 1.0)


def write_test_raster(
    path: Path,
    data: np.ndarray,
    *,
    transform: Affine = DEFAULT_TRANSFORM,
    crs: str = DEFAULT_CRS,
    nodata: Optional[float] = -9999,
) -> Path:
    """Write a small GeoTIFF fixture for tests."""
    array = np.asarray(data)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    if array.ndim != 3:
        raise ValueError("data must be a 2D array or a 3D band-first array")

    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": int(array.shape[1]),
        "width": int(array.shape[2]),
        "count": int(array.shape[0]),
        "dtype": str(array.dtype),
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)
    return path
