"""Chunked pairwise alignment pipeline for split-factor execution."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds

from coregix.preprocess.registration import (
    estimate_elastix_transform,
)

DEFAULT_ALIGNMENT_PARAMETER_MAPS = ["translation", "rigid"]
CHUNK_OVERLAP_PX = 256
ANCHOR_GRID_SIZE = 3


@dataclass
class AlignmentResult:
    output_image_path: str
    temp_dir: Optional[str]


@dataclass
class GlobalRigidTransform:
    rotation: np.ndarray
    translation: np.ndarray


def _to_int_window(window: Window, max_width: int, max_height: int) -> Window:
    col0 = max(0, int(np.floor(window.col_off)))
    row0 = max(0, int(np.floor(window.row_off)))
    col1 = min(max_width, int(np.ceil(window.col_off + window.width)))
    row1 = min(max_height, int(np.ceil(window.row_off + window.height)))
    return Window(col_off=col0, row_off=row0, width=max(0, col1 - col0), height=max(0, row1 - row0))


def _window_in_parent(parent: Window, child: Window) -> Window:
    return Window(
        col_off=int(parent.col_off + child.col_off),
        row_off=int(parent.row_off + child.row_off),
        width=int(child.width),
        height=int(child.height),
    )


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
    profile = src_profile.copy()
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


def _coerce_output_nodata(dtype_name: str, nodata: float) -> float:
    dtype = np.dtype(dtype_name)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        if nodata < info.min or nodata > info.max:
            return float(0)
    return float(nodata)


def _edge_proxy(data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    arr = data.astype(np.float32, copy=True)
    arr[~valid_mask] = 0.0
    gy, gx = np.gradient(arr)
    edge = np.hypot(gx, gy)
    edge[~valid_mask] = 0.0
    return edge.astype(np.float32)


def _sample_bilinear(
    data: np.ndarray,
    row_coords: np.ndarray,
    col_coords: np.ndarray,
    *,
    fill_value: float,
) -> tuple[np.ndarray, np.ndarray]:
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
    cols = np.arange(width, dtype=np.float64) + 0.5
    rows = np.arange(height, dtype=np.float64) + 0.5
    col_grid, row_grid = np.meshgrid(cols, rows)
    x_world = transform.a * col_grid + transform.b * row_grid + transform.c
    y_world = transform.d * col_grid + transform.e * row_grid + transform.f
    return x_world, y_world


def _world_to_array_coords(transform, x_world: np.ndarray, y_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv = ~transform
    center_cols = inv.a * x_world + inv.b * y_world + inv.c
    center_rows = inv.d * x_world + inv.e * y_world + inv.f
    return center_rows - 0.5, center_cols - 0.5


def _array_to_world(transform, row_coords: np.ndarray, col_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center_cols = col_coords + 0.5
    center_rows = row_coords + 0.5
    x_world = transform.a * center_cols + transform.b * center_rows + transform.c
    y_world = transform.d * center_cols + transform.e * center_rows + transform.f
    return x_world, y_world


def _expand_window(window: Window, overlap: int, max_width: int, max_height: int) -> Window:
    return _to_int_window(
        Window(
            col_off=window.col_off - overlap,
            row_off=window.row_off - overlap,
            width=window.width + 2 * overlap,
            height=window.height + 2 * overlap,
        ),
        max_width=max_width,
        max_height=max_height,
    )


def _split_positions(length: int, parts: int) -> list[int]:
    if parts <= 0:
        raise ValueError("parts must be > 0.")
    return [int(i * length // parts) for i in range(parts + 1)]


def _chunk_grid_shape(split_factor: int, width: int, height: int) -> tuple[int, int]:
    if split_factor < 0:
        raise ValueError("split_factor must be >= 0.")
    if split_factor == 0:
        return 1, 1
    major_exp = (split_factor + 1) // 2
    minor_exp = split_factor // 2
    if width >= height:
        return 2**minor_exp, 2**major_exp
    return 2**major_exp, 2**minor_exp


def _resolve_solve_grid(
    *,
    base_transform,
    base_width: int,
    base_height: int,
    solve_resolution: Optional[float],
) -> tuple[int, int, Affine]:
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


def _window_corners_world(window: Window, transform) -> np.ndarray:
    bounds = rasterio.windows.bounds(window, transform)
    return np.array(
        [
            [bounds[0], bounds[1]],
            [bounds[0], bounds[3]],
            [bounds[2], bounds[1]],
            [bounds[2], bounds[3]],
        ],
        dtype=np.float64,
    )


def _transform_points(transform: GlobalRigidTransform, points_xy: np.ndarray) -> np.ndarray:
    return points_xy @ transform.rotation.T + transform.translation


def _apply_world_transform(
    transform: GlobalRigidTransform,
    x_world: np.ndarray,
    y_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_x = (
        transform.rotation[0, 0] * x_world
        + transform.rotation[0, 1] * y_world
        + transform.translation[0]
    )
    source_y = (
        transform.rotation[1, 0] * x_world
        + transform.rotation[1, 1] * y_world
        + transform.translation[1]
    )
    return source_x, source_y


def _fit_rigid_transform(target_xy: np.ndarray, source_xy: np.ndarray) -> GlobalRigidTransform:
    if target_xy.shape[0] < 2 or source_xy.shape[0] < 2:
        raise ValueError("At least two point pairs are required to fit a rigid transform.")
    target_mean = target_xy.mean(axis=0)
    source_mean = source_xy.mean(axis=0)
    target_centered = target_xy - target_mean
    source_centered = source_xy - source_mean
    covariance = target_centered.T @ source_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = source_mean - rotation @ target_mean
    return GlobalRigidTransform(rotation=rotation, translation=translation)


def _fit_global_rigid_transform(target_xy: np.ndarray, source_xy: np.ndarray) -> GlobalRigidTransform:
    if target_xy.shape[0] < 3 or source_xy.shape[0] < 3:
        raise ValueError("At least three point pairs are required for chunked global solve.")
    inliers = np.ones(target_xy.shape[0], dtype=bool)
    transform = _fit_rigid_transform(target_xy, source_xy)
    for _ in range(3):
        predicted = _transform_points(transform, target_xy)
        residuals = np.linalg.norm(predicted - source_xy, axis=1)
        residuals_inliers = residuals[inliers]
        median = float(np.median(residuals_inliers))
        mad = float(np.median(np.abs(residuals_inliers - median)))
        if mad <= 1e-9:
            break
        sigma = 1.4826 * mad
        threshold = max(median + 3.0 * sigma, 1e-6)
        updated_inliers = residuals <= threshold
        if updated_inliers.sum() < 3 or np.array_equal(updated_inliers, inliers):
            break
        inliers = updated_inliers
        transform = _fit_rigid_transform(target_xy[inliers], source_xy[inliers])
    return transform


def _apply_parameter_map_to_points(parameter_map, points_xy: np.ndarray) -> np.ndarray:
    transform_kind = list(parameter_map["Transform"])
    params = np.array([float(value) for value in parameter_map["TransformParameters"]], dtype=np.float64)
    if transform_kind == ["TranslationTransform"]:
        return points_xy + params[None, :]
    if transform_kind == ["EulerTransform"]:
        angle, tx, ty = params
        center = np.array([float(value) for value in parameter_map["CenterOfRotationPoint"]], dtype=np.float64)
        cos_theta = float(np.cos(angle))
        sin_theta = float(np.sin(angle))
        rotation = np.array(
            [[cos_theta, -sin_theta], [sin_theta, cos_theta]],
            dtype=np.float64,
        )
        return (points_xy - center) @ rotation.T + center + np.array([tx, ty], dtype=np.float64)
    raise ValueError(f"Unsupported elastix transform kind in chunked global solve: {transform_kind}")


def _apply_parameter_object_to_points(transform_parameter_object, points_xy: np.ndarray) -> np.ndarray:
    transformed = np.asarray(points_xy, dtype=np.float64)
    for idx in range(int(transform_parameter_object.GetNumberOfParameterMaps())):
        transformed = _apply_parameter_map_to_points(transform_parameter_object.GetParameterMap(idx), transformed)
    return transformed


def _source_window_for_target_window(
    target_window: Window,
    target_transform,
    moving_transform,
    transform: GlobalRigidTransform,
    moving_width: int,
    moving_height: int,
    padding_pixels: int = 2,
) -> Window:
    corners = _window_corners_world(target_window, target_transform)
    source_corners = _transform_points(transform, corners)
    source_window = _to_int_window(
        from_bounds(
            left=float(source_corners[:, 0].min()),
            bottom=float(source_corners[:, 1].min()),
            right=float(source_corners[:, 0].max()),
            top=float(source_corners[:, 1].max()),
            transform=moving_transform,
        ),
        max_width=moving_width,
        max_height=moving_height,
    )
    return _expand_window(source_window, padding_pixels, moving_width, moving_height)


def _estimate_chunk_correspondences(
    *,
    chunk_id: str,
    fixed_src: rasterio.DatasetReader,
    moving_src: rasterio.DatasetReader,
    fixed_window: Window,
    fixed_window_transform,
    moving_band_1based: int,
    fixed_band_1based: int,
    moving_nodata_value: Optional[float],
    fixed_nodata_value: Optional[float],
    out_nodata: float,
    min_valid_fraction: float,
    work_dir: str,
    log_to_console: bool,
    enforce_mutual_valid_mask: bool,
    use_edge_proxies: bool,
    solve_transform,
    core_solve_window: Window,
    chunk_solve_window: Window,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    chunk_solve_transform = rasterio.windows.transform(chunk_solve_window, solve_transform)
    chunk_bounds = rasterio.windows.bounds(
        Window(0, 0, chunk_solve_window.width, chunk_solve_window.height),
        chunk_solve_transform,
    )
    fixed_chunk_rel = _to_int_window(
        from_bounds(
            left=chunk_bounds[0],
            bottom=chunk_bounds[1],
            right=chunk_bounds[2],
            top=chunk_bounds[3],
            transform=fixed_window_transform,
        ),
        max_width=int(fixed_window.width),
        max_height=int(fixed_window.height),
    )
    moving_chunk_abs = _to_int_window(
        from_bounds(
            left=chunk_bounds[0],
            bottom=chunk_bounds[1],
            right=chunk_bounds[2],
            top=chunk_bounds[3],
            transform=moving_src.transform,
        ),
        max_width=moving_src.width,
        max_height=moving_src.height,
    )
    if (
        fixed_chunk_rel.width <= 0
        or fixed_chunk_rel.height <= 0
        or moving_chunk_abs.width <= 0
        or moving_chunk_abs.height <= 0
    ):
        return None

    fixed_chunk_abs = _window_in_parent(fixed_window, fixed_chunk_rel)
    fixed_band = fixed_src.read(fixed_band_1based, window=fixed_chunk_abs)
    moving_band = moving_src.read(moving_band_1based, window=moving_chunk_abs)
    fixed_valid = fixed_src.read_masks(fixed_band_1based, window=fixed_chunk_abs) > 0
    moving_valid = moving_src.read_masks(moving_band_1based, window=moving_chunk_abs) > 0
    if fixed_nodata_value is not None:
        fixed_valid &= fixed_band != fixed_nodata_value
    if moving_nodata_value is not None:
        moving_valid &= moving_band != moving_nodata_value

    solve_height = int(chunk_solve_window.height)
    solve_width = int(chunk_solve_window.width)
    fixed_reg_data = np.full(
        (solve_height, solve_width),
        fixed_nodata_value if fixed_nodata_value is not None else out_nodata,
        dtype=np.float32,
    )
    moving_reg_data = np.full(
        (solve_height, solve_width),
        moving_nodata_value if moving_nodata_value is not None else out_nodata,
        dtype=np.float32,
    )
    fixed_valid_reprojected = np.zeros((solve_height, solve_width), dtype=np.uint8)
    moving_valid_reprojected = np.zeros((solve_height, solve_width), dtype=np.uint8)

    reproject(
        source=fixed_band.astype(np.float32),
        destination=fixed_reg_data,
        src_transform=fixed_src.window_transform(fixed_chunk_abs),
        src_crs=fixed_src.crs,
        dst_transform=chunk_solve_transform,
        dst_crs=fixed_src.crs,
        src_nodata=fixed_nodata_value,
        dst_nodata=fixed_nodata_value if fixed_nodata_value is not None else out_nodata,
        resampling=Resampling.nearest,
    )
    reproject(
        source=moving_band.astype(np.float32),
        destination=moving_reg_data,
        src_transform=moving_src.window_transform(moving_chunk_abs),
        src_crs=moving_src.crs,
        dst_transform=chunk_solve_transform,
        dst_crs=fixed_src.crs,
        src_nodata=moving_nodata_value,
        dst_nodata=moving_nodata_value if moving_nodata_value is not None else out_nodata,
        resampling=Resampling.nearest,
    )
    reproject(
        source=fixed_valid.astype(np.uint8),
        destination=fixed_valid_reprojected,
        src_transform=fixed_src.window_transform(fixed_chunk_abs),
        src_crs=fixed_src.crs,
        dst_transform=chunk_solve_transform,
        dst_crs=fixed_src.crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    reproject(
        source=moving_valid.astype(np.uint8),
        destination=moving_valid_reprojected,
        src_transform=moving_src.window_transform(moving_chunk_abs),
        src_crs=moving_src.crs,
        dst_transform=chunk_solve_transform,
        dst_crs=fixed_src.crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )

    fixed_valid_for_registration = fixed_valid_reprojected > 0
    moving_valid_for_registration = moving_valid_reprojected > 0
    if use_edge_proxies:
        fixed_reg_data = _edge_proxy(fixed_reg_data, fixed_valid_for_registration)
        moving_reg_data = _edge_proxy(moving_reg_data, moving_valid_for_registration)
        fixed_mask_for_elastix = (fixed_reg_data > 0).astype(np.uint8)
        moving_mask_for_elastix = (moving_reg_data > 0).astype(np.uint8)
    else:
        fixed_mask_for_elastix = fixed_valid_for_registration.astype(np.uint8)
        moving_mask_for_elastix = moving_valid_for_registration.astype(np.uint8)
    if enforce_mutual_valid_mask:
        mutual = (fixed_mask_for_elastix > 0) & (moving_mask_for_elastix > 0)
        fixed_mask_for_elastix = mutual.astype(np.uint8)
        moving_mask_for_elastix = mutual.astype(np.uint8)

    min_valid_pixels = int(max(1, min_valid_fraction * (solve_width * solve_height)))
    if int(fixed_mask_for_elastix.sum()) < min_valid_pixels:
        return None
    if int(moving_mask_for_elastix.sum()) < min_valid_pixels:
        return None

    fixed_reg_path = os.path.join(work_dir, f"{chunk_id}_fixed_reg.tif")
    moving_reg_path = os.path.join(work_dir, f"{chunk_id}_moving_reg.tif")
    fixed_mask_path = os.path.join(work_dir, f"{chunk_id}_fixed_mask.tif")
    moving_mask_path = os.path.join(work_dir, f"{chunk_id}_moving_mask.tif")
    _write_single_band_tif(
        fixed_reg_path,
        fixed_reg_data.astype("float32"),
        crs=fixed_src.crs,
        transform=chunk_solve_transform,
        dtype="float32",
        nodata=fixed_nodata_value,
    )
    _write_single_band_tif(
        moving_reg_path,
        moving_reg_data.astype("float32"),
        crs=fixed_src.crs,
        transform=chunk_solve_transform,
        dtype="float32",
        nodata=moving_nodata_value,
    )
    _write_single_band_tif(
        fixed_mask_path,
        fixed_mask_for_elastix.astype("uint8"),
        crs=fixed_src.crs,
        transform=chunk_solve_transform,
        dtype="uint8",
        nodata=0,
    )
    _write_single_band_tif(
        moving_mask_path,
        moving_mask_for_elastix.astype("uint8"),
        crs=fixed_src.crs,
        transform=chunk_solve_transform,
        dtype="uint8",
        nodata=0,
    )

    try:
        transform_parameter_object = estimate_elastix_transform(
            fixed_image_path=fixed_reg_path,
            moving_image_path=moving_reg_path,
            parameter_map=DEFAULT_ALIGNMENT_PARAMETER_MAPS,
            force_nearest_resample=True,
            fixed_mask_path=fixed_mask_path,
            moving_mask_path=moving_mask_path,
            log_to_console=log_to_console,
        )
    except Exception:
        return None

    local_core_row0 = float(core_solve_window.row_off - chunk_solve_window.row_off)
    local_core_col0 = float(core_solve_window.col_off - chunk_solve_window.col_off)
    row_centers = local_core_row0 + (
        (np.arange(ANCHOR_GRID_SIZE, dtype=np.float64) + 0.5) * (float(core_solve_window.height) / ANCHOR_GRID_SIZE)
    ) - 0.5
    col_centers = local_core_col0 + (
        (np.arange(ANCHOR_GRID_SIZE, dtype=np.float64) + 0.5) * (float(core_solve_window.width) / ANCHOR_GRID_SIZE)
    ) - 0.5
    anchor_rows, anchor_cols = np.meshgrid(row_centers, col_centers, indexing="ij")
    source_points_local = _apply_parameter_object_to_points(
        transform_parameter_object,
        np.column_stack([anchor_cols.ravel(), anchor_rows.ravel()]),
    )
    target_x, target_y = _array_to_world(chunk_solve_transform, anchor_rows, anchor_cols)
    source_x, source_y = _array_to_world(
        chunk_solve_transform,
        source_points_local[:, 1].reshape(anchor_rows.shape),
        source_points_local[:, 0].reshape(anchor_cols.shape),
    )
    target_points = np.column_stack([target_x.ravel(), target_y.ravel()])
    source_points = np.column_stack([source_x.ravel(), source_y.ravel()])
    if not np.isfinite(source_points).all():
        return None
    return target_points, source_points


def align_image_pair(
    moving_image_path: str,
    fixed_image_path: str,
    output_image_path: str,
    *,
    band_index: int = 0,
    moving_band_index: Optional[int] = None,
    fixed_band_index: Optional[int] = None,
    moving_nodata: Optional[float] = None,
    fixed_nodata: Optional[float] = None,
    output_nodata: Optional[float] = None,
    min_valid_fraction: float = 0.01,
    temp_dir: Optional[str] = None,
    keep_temp_dir: bool = False,
    log_to_console: bool = False,
    clip_fixed_to_moving: bool = False,
    output_on_moving_grid: bool = True,
    trim_edge_invalid: bool = False,
    edge_trim_depth: int = 8,
    edge_trim_detection_band_index: int = 0,
    edge_trim_invalid_below: Optional[float] = None,
    edge_trim_invalid_above: Optional[float] = None,
    enforce_mutual_valid_mask: bool = False,
    use_edge_proxies: bool = True,
    split_factor: int = 2,
    solve_resolution: Optional[float] = None,
) -> AlignmentResult:
    if band_index < 0:
        raise ValueError("band_index must be >= 0 (0-based).")
    if moving_band_index is not None and moving_band_index < 0:
        raise ValueError("moving_band_index must be >= 0 (0-based).")
    if fixed_band_index is not None and fixed_band_index < 0:
        raise ValueError("fixed_band_index must be >= 0 (0-based).")
    if min_valid_fraction <= 0 or min_valid_fraction > 1:
        raise ValueError("min_valid_fraction must be in (0, 1].")
    if edge_trim_depth <= 0:
        raise ValueError("edge_trim_depth must be > 0.")
    if edge_trim_detection_band_index < 0:
        raise ValueError("edge_trim_detection_band_index must be >= 0.")
    if split_factor <= 0:
        raise ValueError("split_factor must be > 0 for the chunked alignment path.")
    if solve_resolution is not None and solve_resolution <= 0:
        raise ValueError("solve_resolution must be > 0 when provided.")
    temp_ctx = None
    work_dir: str
    if keep_temp_dir:
        work_dir = tempfile.mkdtemp(prefix="vhr_align_", dir=temp_dir)
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="vhr_align_", dir=temp_dir)
        work_dir = temp_ctx.name

    with rasterio.open(fixed_image_path) as fixed_src, rasterio.open(moving_image_path) as moving_src:
        moving_band_1based = (moving_band_index if moving_band_index is not None else band_index) + 1
        fixed_band_1based = (fixed_band_index if fixed_band_index is not None else band_index) + 1
        if fixed_band_1based > fixed_src.count:
            raise ValueError(
                f"Requested fixed band index={fixed_band_1based - 1}, but fixed image has {fixed_src.count} band(s)."
            )
        if moving_band_1based > moving_src.count:
            raise ValueError(
                f"Requested moving band index={moving_band_1based - 1}, but moving image has {moving_src.count} band(s)."
            )
        if fixed_src.crs is None or moving_src.crs is None:
            raise ValueError("Both fixed and moving images must have CRS.")
        if fixed_src.crs != moving_src.crs:
            raise ValueError(
                "Fixed and moving images must share the same CRS for tile-window extraction. "
                f"fixed={fixed_src.crs}, moving={moving_src.crs}"
            )

        moving_nodata_value = _resolve_nodata(moving_src, moving_nodata)
        fixed_nodata_value = _resolve_nodata(fixed_src, fixed_nodata)
        out_nodata = (
            output_nodata
            if output_nodata is not None
            else moving_nodata_value
            if moving_nodata_value is not None
            else fixed_nodata_value
            if fixed_nodata_value is not None
            else 0
        )
        out_nodata = _coerce_output_nodata(moving_src.dtypes[0], float(out_nodata))

        if clip_fixed_to_moving:
            fixed_window = _to_int_window(
                from_bounds(
                    left=moving_src.bounds.left,
                    bottom=moving_src.bounds.bottom,
                    right=moving_src.bounds.right,
                    top=moving_src.bounds.top,
                    transform=fixed_src.transform,
                ),
                max_width=fixed_src.width,
                max_height=fixed_src.height,
            )
            if fixed_window.width <= 0 or fixed_window.height <= 0:
                raise ValueError("No overlap between moving-image bounds and fixed-image grid.")
        else:
            fixed_window = Window(col_off=0, row_off=0, width=fixed_src.width, height=fixed_src.height)

        fixed_window_transform = fixed_src.window_transform(fixed_window)
        fixed_bounds = fixed_src.window_bounds(fixed_window)
        moving_window = _to_int_window(
            from_bounds(
                left=fixed_bounds[0],
                bottom=fixed_bounds[1],
                right=fixed_bounds[2],
                top=fixed_bounds[3],
                transform=moving_src.transform,
            ),
            max_width=moving_src.width,
            max_height=moving_src.height,
        )
        if moving_window.width <= 0 or moving_window.height <= 0:
            raise ValueError("No overlap between fixed-image ROI and moving-image grid.")
        moving_window_transform = moving_src.window_transform(moving_window)

        solve_width, solve_height, solve_transform = _resolve_solve_grid(
            base_transform=fixed_window_transform,
            base_width=int(fixed_window.width),
            base_height=int(fixed_window.height),
            solve_resolution=solve_resolution,
        )

        solve_rows, solve_cols = _chunk_grid_shape(split_factor, solve_width, solve_height)
        solve_row_splits = _split_positions(solve_height, solve_rows)
        solve_col_splits = _split_positions(solve_width, solve_cols)

        target_points: list[np.ndarray] = []
        source_points: list[np.ndarray] = []
        for row_idx in range(solve_rows):
            for col_idx in range(solve_cols):
                core_solve_window = Window(
                    col_off=solve_col_splits[col_idx],
                    row_off=solve_row_splits[row_idx],
                    width=solve_col_splits[col_idx + 1] - solve_col_splits[col_idx],
                    height=solve_row_splits[row_idx + 1] - solve_row_splits[row_idx],
                )
                if core_solve_window.width <= 0 or core_solve_window.height <= 0:
                    continue
                chunk_solve_window = _expand_window(
                    core_solve_window,
                    CHUNK_OVERLAP_PX,
                    solve_width,
                    solve_height,
                )
                chunk_pairs = _estimate_chunk_correspondences(
                    chunk_id=f"chunk_r{row_idx}_c{col_idx}",
                    fixed_src=fixed_src,
                    moving_src=moving_src,
                    fixed_window=fixed_window,
                    fixed_window_transform=fixed_window_transform,
                    moving_band_1based=moving_band_1based,
                    fixed_band_1based=fixed_band_1based,
                    moving_nodata_value=moving_nodata_value,
                    fixed_nodata_value=fixed_nodata_value,
                    out_nodata=out_nodata,
                    min_valid_fraction=min_valid_fraction,
                    work_dir=work_dir,
                    log_to_console=log_to_console,
                    enforce_mutual_valid_mask=enforce_mutual_valid_mask,
                    use_edge_proxies=use_edge_proxies,
                    solve_transform=solve_transform,
                    core_solve_window=core_solve_window,
                    chunk_solve_window=chunk_solve_window,
                )
                if chunk_pairs is None:
                    continue
                target_chunk_points, source_chunk_points = chunk_pairs
                target_points.append(target_chunk_points)
                source_points.append(source_chunk_points)

        if not target_points:
            raise ValueError("No chunk produced a valid local solve. Try a lower split_factor or coarser solve_resolution.")
        global_transform = _fit_global_rigid_transform(
            np.vstack(target_points),
            np.vstack(source_points),
        )

        os.makedirs(os.path.dirname(output_image_path) or ".", exist_ok=True)
        temp_output_image_path = os.path.join(work_dir, os.path.basename(output_image_path))
        if output_on_moving_grid:
            out_profile = _make_output_profile(
                moving_src.profile,
                count=moving_src.count,
                dtype=moving_src.dtypes[0],
                nodata=out_nodata,
            )
            target_rows, target_cols = _chunk_grid_shape(split_factor, int(moving_window.width), int(moving_window.height))
            target_row_splits = _split_positions(int(moving_window.height), target_rows)
            target_col_splits = _split_positions(int(moving_window.width), target_cols)
        else:
            out_profile = _make_output_profile(
                fixed_src.profile,
                count=moving_src.count,
                dtype=moving_src.dtypes[0],
                nodata=out_nodata,
                width=int(fixed_window.width),
                height=int(fixed_window.height),
                transform=fixed_window_transform,
            )
            target_rows, target_cols = _chunk_grid_shape(split_factor, int(fixed_window.width), int(fixed_window.height))
            target_row_splits = _split_positions(int(fixed_window.height), target_rows)
            target_col_splits = _split_positions(int(fixed_window.width), target_cols)

        with rasterio.open(temp_output_image_path, "w+", **out_profile) as out_dst:
            try:
                out_dst.colorinterp = moving_src.colorinterp
            except Exception:
                pass
            try:
                out_dst.scales = moving_src.scales
                out_dst.offsets = moving_src.offsets
            except Exception:
                pass
            for b in range(1, moving_src.count + 1):
                desc = moving_src.descriptions[b - 1]
                if desc:
                    out_dst.set_band_description(b, desc)
                band_tags = moving_src.tags(b).copy()
                for k in list(band_tags.keys()):
                    if k.upper().startswith("STATISTICS_"):
                        band_tags.pop(k, None)
                out_dst.update_tags(b, **band_tags)
                out_dst.update_tags(
                    b,
                    STATISTICS_MINIMUM="",
                    STATISTICS_MAXIMUM="",
                    STATISTICS_MEAN="",
                    STATISTICS_STDDEV="",
                )

            if output_on_moving_grid:
                for b in range(1, moving_src.count + 1):
                    for _, block_window in out_dst.block_windows(b):
                        out_dst.write(
                            moving_src.read(b, window=block_window).astype(out_profile["dtype"]),
                            b,
                            window=block_window,
                        )
            else:
                for b in range(1, moving_src.count + 1):
                    for _, block_window in out_dst.block_windows(b):
                        out_dst.write(
                            np.full(
                                (int(block_window.height), int(block_window.width)),
                                out_nodata,
                                dtype=out_profile["dtype"],
                            ),
                            b,
                            window=block_window,
                        )

            for b in range(1, moving_src.count + 1):
                block_width, block_height = out_dst.block_shapes[b - 1]
                for row_idx in range(target_rows):
                    for col_idx in range(target_cols):
                        core_window = Window(
                            col_off=target_col_splits[col_idx],
                            row_off=target_row_splits[row_idx],
                            width=target_col_splits[col_idx + 1] - target_col_splits[col_idx],
                            height=target_row_splits[row_idx + 1] - target_row_splits[row_idx],
                        )
                        if core_window.width <= 0 or core_window.height <= 0:
                            continue
                        if output_on_moving_grid:
                            target_world_transform = moving_src.transform
                            target_abs_window = _window_in_parent(moving_window, core_window)
                            target_out_window = target_abs_window
                        else:
                            target_world_transform = fixed_src.transform
                            target_abs_window = _window_in_parent(fixed_window, core_window)
                            target_out_window = core_window
                        source_window = _source_window_for_target_window(
                            target_abs_window,
                            target_world_transform,
                            moving_src.transform,
                            global_transform,
                            moving_src.width,
                            moving_src.height,
                        )
                        if source_window.width <= 0 or source_window.height <= 0:
                            continue
                        moving_band_data = moving_src.read(b, window=source_window).astype(np.float32)
                        moving_band_valid = moving_src.read_masks(b, window=source_window).astype(np.float32)
                        if moving_nodata_value is not None:
                            moving_band_valid *= (moving_band_data != moving_nodata_value).astype(np.float32)
                        source_transform = moving_src.window_transform(source_window)

                        core_row0 = int(core_window.row_off)
                        core_col0 = int(core_window.col_off)
                        core_row1 = core_row0 + int(core_window.height)
                        core_col1 = core_col0 + int(core_window.width)
                        for row_off in range(core_row0, core_row1, int(block_height)):
                            for col_off in range(core_col0, core_col1, int(block_width)):
                                win_w = min(int(block_width), core_col1 - col_off)
                                win_h = min(int(block_height), core_row1 - row_off)
                                block_window = Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)
                                if output_on_moving_grid:
                                    target_block_abs = _window_in_parent(moving_window, block_window)
                                    target_block_out = target_block_abs
                                else:
                                    target_block_abs = _window_in_parent(fixed_window, block_window)
                                    target_block_out = block_window
                                block_transform = rasterio.windows.transform(target_block_abs, target_world_transform)
                                x_world, y_world = _pixel_centers_world(block_transform, win_h, win_w)
                                solve_rows_block, solve_cols_block = _world_to_array_coords(
                                    solve_transform,
                                    x_world,
                                    y_world,
                                )
                                target_valid = (
                                    (solve_rows_block >= 0.0)
                                    & (solve_rows_block <= solve_height - 1)
                                    & (solve_cols_block >= 0.0)
                                    & (solve_cols_block <= solve_width - 1)
                                )
                                source_x_world, source_y_world = _apply_world_transform(
                                    global_transform,
                                    x_world,
                                    y_world,
                                )
                                source_rows, source_cols = _world_to_array_coords(
                                    source_transform,
                                    source_x_world,
                                    source_y_world,
                                )
                                remapped_block, moving_valid = _sample_bilinear(
                                    moving_band_data,
                                    source_rows,
                                    source_cols,
                                    fill_value=out_nodata,
                                )
                                sampled_mask, mask_valid = _sample_bilinear(
                                    moving_band_valid,
                                    source_rows,
                                    source_cols,
                                    fill_value=0.0,
                                )
                                valid = target_valid & moving_valid & mask_valid & (sampled_mask > 0.0)
                                combined = np.where(valid, remapped_block, out_nodata)
                                out_dst.write(
                                    combined.astype(out_profile["dtype"]),
                                    b,
                                    window=target_block_out,
                                )

        if trim_edge_invalid:
            from coregix.postprocess import trim_edge_invalid_pixels

            trim_edge_invalid_pixels(
                input_image_path=temp_output_image_path,
                output_image_path=output_image_path,
                edge_depth=edge_trim_depth,
                detection_band_index=edge_trim_detection_band_index,
                invalid_below=edge_trim_invalid_below,
                invalid_above=edge_trim_invalid_above,
                nodata_value=out_nodata,
            )
        else:
            shutil.copyfile(temp_output_image_path, output_image_path)

    if temp_ctx is not None:
        temp_ctx.cleanup()
        kept_temp = None
    else:
        kept_temp = work_dir

    return AlignmentResult(
        output_image_path=output_image_path,
        temp_dir=kept_temp,
    )
