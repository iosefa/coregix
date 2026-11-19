"""Joint raster alignment and retained array/grid helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window

from coregix._joint.joint_coregistration.joint_coregistration import align_image_pair


def _to_int_window(window: Window, max_width: int, max_height: int) -> Window:
    col0 = max(0, int(np.floor(window.col_off)))
    row0 = max(0, int(np.floor(window.row_off)))
    col1 = min(max_width, int(np.ceil(window.col_off + window.width)))
    row1 = min(max_height, int(np.ceil(window.row_off + window.height)))
    return Window(col_off=col0, row_off=row0, width=max(0, col1 - col0), height=max(0, row1 - row0))


def _write_single_band_tif(
    path: str,
    data: np.ndarray,
    *,
    crs,
    transform,
    dtype: str,
    nodata: Optional[float],
) -> None:
    profile = {
        "driver": "GTiff",
        "count": 1,
        "height": int(data.shape[0]),
        "width": int(data.shape[1]),
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "bigtiff": "yes",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _stream_reprojected_band_to_tif(
    path: str,
    *,
    src: rasterio.DatasetReader,
    band_index: int,
    dst_crs,
    dst_transform,
    dst_width: int,
    dst_height: int,
    src_nodata: Optional[float],
    dst_fill_value: float,
    output_nodata: Optional[float],
) -> None:
    """Reproject a source band into a temporary TIFF without materializing the full band."""
    profile = {
        "driver": "GTiff",
        "count": 1,
        "height": int(dst_height),
        "width": int(dst_width),
        "dtype": "float32",
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": output_nodata,
        "bigtiff": "yes",
    }
    with rasterio.open(path, "w", **profile) as dst:
        for _, block_window in dst.block_windows(1):
            block = np.full(
                (int(block_window.height), int(block_window.width)),
                dst_fill_value,
                dtype=np.float32,
            )
            reproject(
                source=rasterio.band(src, band_index),
                destination=block,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=rasterio.windows.transform(block_window, dst_transform),
                dst_crs=dst_crs,
                dst_nodata=dst_fill_value,
                resampling=Resampling.nearest,
            )
            dst.write(block, 1, window=block_window)


def _make_output_profile(
    src_profile: dict,
    *,
    count: int,
    dtype: str,
    nodata: float,
    width: Optional[int] = None,
    height: Optional[int] = None,
    transform=None,
) -> dict:
    """Build a GeoTIFF output profile without inheriting invalid tiling metadata."""
    profile = src_profile.copy()

    # Source profiles can carry nonstandard block sizes that are valid for reading
    # but rejected when reused as GeoTIFF creation options for a new file.
    for key in ("blockxsize", "blockysize"):
        profile.pop(key, None)

    profile.update(
        count=count,
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        predictor=2,
        tiled=True,
        interleave="band",
        BIGTIFF="IF_SAFER",
    )
    if width is not None:
        profile["width"] = int(width)
    if height is not None:
        profile["height"] = int(height)
    if transform is not None:
        profile["transform"] = transform
    return profile


def _resolve_nodata(src: rasterio.DatasetReader, override_nodata: Optional[float]) -> Optional[float]:
    return override_nodata if override_nodata is not None else src.nodata


def _edge_proxy(data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    arr = data.astype(np.float32, copy=False)
    valid = valid_mask.astype(bool, copy=False)
    edge = np.zeros(arr.shape, dtype=np.float32)
    if not valid.any():
        return edge

    gx = np.zeros(arr.shape, dtype=np.float32)
    gy = np.zeros(arr.shape, dtype=np.float32)

    horizontal_valid = valid[:, 1:-1] & valid[:, :-2] & valid[:, 2:]
    gx_mid = gx[:, 1:-1]
    gx_mid[horizontal_valid] = 0.5 * (
        arr[:, 2:][horizontal_valid] - arr[:, :-2][horizontal_valid]
    )

    vertical_valid = valid[1:-1, :] & valid[:-2, :] & valid[2:, :]
    gy_mid = gy[1:-1, :]
    gy_mid[vertical_valid] = 0.5 * (
        arr[2:, :][vertical_valid] - arr[:-2, :][vertical_valid]
    )

    edge = np.hypot(gx, gy).astype(np.float32)
    edge[~valid] = 0.0
    return edge.astype(np.float32)


def _sample_bilinear(
    data: np.ndarray,
    row_coords: np.ndarray,
    col_coords: np.ndarray,
    *,
    fill_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a 2D array at floating-point row/col coordinates."""
    height, width = data.shape
    valid = (
        (row_coords >= 0.0)
        & (row_coords <= height - 1)
        & (col_coords >= 0.0)
        & (col_coords <= width - 1)
    )
    row0 = np.floor(row_coords).astype(np.int64)
    col0 = np.floor(col_coords).astype(np.int64)
    row1 = np.clip(row0 + 1, 0, height - 1)
    col1 = np.clip(col0 + 1, 0, width - 1)
    row0 = np.clip(row0, 0, height - 1)
    col0 = np.clip(col0, 0, width - 1)

    row_weight = row_coords - row0
    col_weight = col_coords - col0

    sampled = (
        (1.0 - row_weight) * (1.0 - col_weight) * data[row0, col0]
        + (1.0 - row_weight) * col_weight * data[row0, col1]
        + row_weight * (1.0 - col_weight) * data[row1, col0]
        + row_weight * col_weight * data[row1, col1]
    ).astype(np.float32)
    sampled[~valid] = fill_value
    return sampled, valid


def _pixel_centers_world(transform, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Return world coordinates for pixel centers in a block/window."""
    cols = np.arange(width, dtype=np.float64) + 0.5
    rows = np.arange(height, dtype=np.float64) + 0.5
    col_grid, row_grid = np.meshgrid(cols, rows)
    x_world = transform.a * col_grid + transform.b * row_grid + transform.c
    y_world = transform.d * col_grid + transform.e * row_grid + transform.f
    return x_world, y_world


def _world_to_array_coords(transform, x_world: np.ndarray, y_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert world coordinates to zero-based array row/col coordinates."""
    inv = ~transform
    center_cols = inv.a * x_world + inv.b * y_world + inv.c
    center_rows = inv.d * x_world + inv.e * y_world + inv.f
    return center_rows - 0.5, center_cols - 0.5


def _array_to_world(transform, row_coords: np.ndarray, col_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert zero-based array row/col coordinates to world coordinates."""
    center_cols = col_coords + 0.5
    center_rows = row_coords + 0.5
    x_world = transform.a * center_cols + transform.b * center_rows + transform.c
    y_world = transform.d * center_cols + transform.e * center_rows + transform.f
    return x_world, y_world


def _resolve_solve_grid(
    *,
    base_transform,
    base_width: int,
    base_height: int,
    solve_resolution: Optional[float],
) -> tuple[int, int, Affine]:
    """Return the registration solve grid dimensions and transform."""
    if solve_resolution is None:
        return base_width, base_height, base_transform

    x_res = float(np.hypot(base_transform.a, base_transform.d))
    y_res = float(np.hypot(base_transform.b, base_transform.e))
    if x_res <= 0.0 or y_res <= 0.0:
        raise ValueError("Unable to determine base raster resolution for solve grid.")

    scale_x = max(1.0, solve_resolution / x_res)
    scale_y = max(1.0, solve_resolution / y_res)
    solve_width = max(1, int(np.ceil(base_width / scale_x)))
    solve_height = max(1, int(np.ceil(base_height / scale_y)))
    solve_transform = base_transform * Affine.scale(
        base_width / solve_width,
        base_height / solve_height,
    )
    return solve_width, solve_height, solve_transform


__all__ = ["align_image_pair"]
