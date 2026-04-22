"""Pairwise alignment pipeline copied from main for large-raster mode."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds

from coregix.preprocess.registration import (
    apply_elastix_transform_subprocess,
    deformation_field_from_transform,
    estimate_elastix_transform,
    write_transform_parameter_files,
)

DEFAULT_ALIGNMENT_PARAMETER_MAPS = ["translation", "rigid"]
LARGE_RASTER_QUADRANT_OVERLAP_PX = 256


@dataclass
class AlignmentResult:
    output_image_path: str
    temp_dir: Optional[str]


@dataclass
class QuadrantChunk:
    core_local_window: Window
    chunk_local_window: Window
    core_source_window: Window
    chunk_source_window: Window
    chunk_transform: object
    fixed_chunk_window: Window
    fixed_chunk_transform: object
    fixed_dx: np.ndarray
    fixed_dy: np.ndarray


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


def align_image_pair(
    moving_image_path: str,
    fixed_image_path: str,
    output_image_path: str,
    *,
    band_index: int = 0,
    moving_band_index: Optional[int] = None,
    fixed_band_index: Optional[int] = None,
    parameter_file_paths: Optional[List[str]] = None,
    moving_nodata: Optional[float] = None,
    fixed_nodata: Optional[float] = None,
    output_nodata: Optional[float] = None,
    min_valid_fraction: float = 0.01,
    temp_dir: Optional[str] = None,
    keep_temp_dir: bool = False,
    log_to_console: bool = False,
    clip_fixed_to_moving: bool = False,
    output_on_moving_grid: bool = True,
    enforce_mutual_valid_mask: bool = False,
    use_edge_proxies: bool = True,
) -> AlignmentResult:
    if band_index < 0:
        raise ValueError("band_index must be >= 0 (0-based).")
    if moving_band_index is not None and moving_band_index < 0:
        raise ValueError("moving_band_index must be >= 0 (0-based).")
    if fixed_band_index is not None and fixed_band_index < 0:
        raise ValueError("fixed_band_index must be >= 0 (0-based).")
    if min_valid_fraction <= 0 or min_valid_fraction > 1:
        raise ValueError("min_valid_fraction must be in (0, 1].")

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
            fixed_domain_window = _to_int_window(
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
            if fixed_domain_window.width <= 0 or fixed_domain_window.height <= 0:
                raise ValueError("No overlap between moving-image bounds and fixed-image grid.")
        else:
            fixed_domain_window = Window(
                col_off=0, row_off=0, width=fixed_src.width, height=fixed_src.height
            )

        os.makedirs(os.path.dirname(output_image_path) or ".", exist_ok=True)
        if output_on_moving_grid:
            out_profile = _make_output_profile(
                moving_src.profile,
                count=moving_src.count,
                dtype=moving_src.dtypes[0],
                nodata=out_nodata,
            )
        else:
            out_profile = _make_output_profile(
                fixed_src.profile,
                count=moving_src.count,
                dtype=moving_src.dtypes[0],
                nodata=out_nodata,
                width=int(fixed_domain_window.width),
                height=int(fixed_domain_window.height),
                transform=fixed_src.window_transform(fixed_domain_window),
            )

        fixed_window = fixed_domain_window
        core_out_window = Window(
            col_off=0,
            row_off=0,
            width=int(fixed_domain_window.width),
            height=int(fixed_domain_window.height),
        )
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

        with rasterio.open(output_image_path, "w+", **out_profile) as out_dst:
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
                        src_block = moving_src.read(b, window=block_window)
                        out_dst.write(src_block.astype(out_profile["dtype"]), b, window=block_window)
            else:
                for b in range(1, moving_src.count + 1):
                    for _, block_window in out_dst.block_windows(b):
                        fill = np.full(
                            (int(block_window.height), int(block_window.width)),
                            out_nodata,
                            dtype=out_profile["dtype"],
                        )
                        out_dst.write(fill, b, window=block_window)

            fixed_band = fixed_src.read(fixed_band_1based, window=fixed_window)
            moving_band = moving_src.read(moving_band_1based, window=moving_window)
            fixed_valid = fixed_src.read_masks(fixed_band_1based, window=fixed_window) > 0
            moving_valid = moving_src.read_masks(moving_band_1based, window=moving_window) > 0
            if fixed_nodata_value is not None:
                fixed_valid &= fixed_band != fixed_nodata_value
            if moving_nodata_value is not None:
                moving_valid &= moving_band != moving_nodata_value

            moving_valid_reprojected = np.zeros(
                (int(fixed_window.height), int(fixed_window.width)),
                dtype=np.uint8,
            )
            reproject(
                source=moving_valid.astype(np.uint8),
                destination=moving_valid_reprojected,
                src_transform=moving_src.window_transform(moving_window),
                src_crs=moving_src.crs,
                dst_transform=fixed_src.window_transform(fixed_window),
                dst_crs=fixed_src.crs,
                src_nodata=0,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )

            fixed_reg_data = fixed_band.astype(np.float32)
            moving_on_fixed = np.full(
                (int(fixed_window.height), int(fixed_window.width)),
                moving_nodata_value if moving_nodata_value is not None else out_nodata,
                dtype=np.float32,
            )
            reproject(
                source=moving_band.astype(np.float32),
                destination=moving_on_fixed,
                src_transform=moving_src.window_transform(moving_window),
                src_crs=moving_src.crs,
                dst_transform=fixed_src.window_transform(fixed_window),
                dst_crs=fixed_src.crs,
                src_nodata=moving_nodata_value,
                dst_nodata=moving_nodata_value if moving_nodata_value is not None else out_nodata,
                resampling=Resampling.nearest,
            )
            moving_on_fixed_valid = moving_valid_reprojected > 0
            if use_edge_proxies:
                fixed_reg_data = _edge_proxy(fixed_reg_data, fixed_valid)
                moving_reg_data = _edge_proxy(moving_on_fixed, moving_on_fixed_valid)
                fixed_mask_for_elastix = (fixed_reg_data > 0).astype(np.uint8)
                moving_mask_for_elastix = (moving_reg_data > 0).astype(np.uint8)
            else:
                moving_reg_data = moving_on_fixed
                fixed_mask_for_elastix = fixed_valid.astype(np.uint8)
                moving_mask_for_elastix = moving_on_fixed_valid.astype(np.uint8)
            if enforce_mutual_valid_mask:
                mutual = (fixed_mask_for_elastix > 0) & (moving_mask_for_elastix > 0)
                fixed_mask_for_elastix = mutual.astype(np.uint8)
                moving_mask_for_elastix = mutual.astype(np.uint8)

            min_valid_pixels = int(max(1, min_valid_fraction * (fixed_window.width * fixed_window.height)))
            if int(fixed_mask_for_elastix.sum()) < min_valid_pixels:
                raise ValueError("Insufficient valid fixed-image support in the registration ROI.")
            if int(moving_mask_for_elastix.sum()) < min_valid_pixels:
                raise ValueError("Insufficient valid moving-image support in the registration ROI.")

            fixed_reg_path = os.path.join(work_dir, "fixed_reg.tif")
            moving_reg_path = os.path.join(work_dir, "moving_reg.tif")
            fixed_mask_path = os.path.join(work_dir, "fixed_mask.tif")
            moving_mask_path = os.path.join(work_dir, "moving_mask.tif")

            _write_single_band_tif(
                fixed_reg_path,
                fixed_reg_data.astype("float32"),
                crs=fixed_src.crs,
                transform=fixed_src.window_transform(fixed_window),
                dtype="float32",
                nodata=fixed_nodata_value,
            )
            _write_single_band_tif(
                moving_reg_path,
                moving_reg_data.astype("float32"),
                crs=fixed_src.crs,
                transform=fixed_src.window_transform(fixed_window),
                dtype="float32",
                nodata=moving_nodata_value,
            )
            _write_single_band_tif(
                fixed_mask_path,
                fixed_mask_for_elastix.astype("uint8"),
                crs=fixed_src.crs,
                transform=fixed_src.window_transform(fixed_window),
                dtype="uint8",
                nodata=0,
            )
            _write_single_band_tif(
                moving_mask_path,
                moving_mask_for_elastix.astype("uint8"),
                crs=fixed_src.crs,
                transform=fixed_src.window_transform(fixed_window),
                dtype="uint8",
                nodata=0,
            )

            try:
                transform_parameter_object = estimate_elastix_transform(
                    fixed_image_path=fixed_reg_path,
                    moving_image_path=moving_reg_path,
                    parameter_map=DEFAULT_ALIGNMENT_PARAMETER_MAPS,
                    parameter_file_paths=parameter_file_paths,
                    force_nearest_resample=True,
                    fixed_mask_path=fixed_mask_path,
                    moving_mask_path=moving_mask_path,
                    log_to_console=log_to_console,
                )
            except Exception as exc:
                raise RuntimeError(f"Elastix registration failed: {exc}") from exc

            quadrant_chunks: list[QuadrantChunk] = []
            if output_on_moving_grid:
                moving_h = int(moving_window.height)
                moving_w = int(moving_window.width)
                row_splits = [0, moving_h // 2, moving_h]
                col_splits = [0, moving_w // 2, moving_w]
                for row_idx in range(2):
                    for col_idx in range(2):
                        core_local_window = Window(
                            col_off=col_splits[col_idx],
                            row_off=row_splits[row_idx],
                            width=col_splits[col_idx + 1] - col_splits[col_idx],
                            height=row_splits[row_idx + 1] - row_splits[row_idx],
                        )
                        if core_local_window.width <= 0 or core_local_window.height <= 0:
                            continue
                        chunk_local_window = _expand_window(
                            core_local_window,
                            LARGE_RASTER_QUADRANT_OVERLAP_PX,
                            moving_w,
                            moving_h,
                        )
                        chunk_source_window = Window(
                            col_off=int(moving_window.col_off + chunk_local_window.col_off),
                            row_off=int(moving_window.row_off + chunk_local_window.row_off),
                            width=int(chunk_local_window.width),
                            height=int(chunk_local_window.height),
                        )
                        core_source_window = Window(
                            col_off=int(moving_window.col_off + core_local_window.col_off),
                            row_off=int(moving_window.row_off + core_local_window.row_off),
                            width=int(core_local_window.width),
                            height=int(core_local_window.height),
                        )
                        chunk_transform = moving_src.window_transform(chunk_source_window)
                        chunk_bounds = rasterio.windows.bounds(
                            Window(0, 0, chunk_source_window.width, chunk_source_window.height),
                            chunk_transform,
                        )
                        fixed_chunk_window = _to_int_window(
                            from_bounds(
                                left=chunk_bounds[0],
                                bottom=chunk_bounds[1],
                                right=chunk_bounds[2],
                                top=chunk_bounds[3],
                                transform=fixed_src.window_transform(fixed_window),
                            ),
                            max_width=int(fixed_window.width),
                            max_height=int(fixed_window.height),
                        )
                        fixed_chunk_window = _expand_window(
                            fixed_chunk_window,
                            2,
                            int(fixed_window.width),
                            int(fixed_window.height),
                        )
                        fixed_chunk_transform = rasterio.windows.transform(
                            fixed_chunk_window,
                            fixed_src.window_transform(fixed_window),
                        )
                        fixed_chunk_ref_path = os.path.join(
                            work_dir,
                            f"fixed_chunk_ref_r{row_idx}_c{col_idx}.tif",
                        )
                        _write_single_band_tif(
                            fixed_chunk_ref_path,
                            np.zeros((int(fixed_chunk_window.height), int(fixed_chunk_window.width)), dtype=np.float32),
                            crs=fixed_src.crs,
                            transform=fixed_chunk_transform,
                            dtype="float32",
                            nodata=0.0,
                        )
                        deformation_field = deformation_field_from_transform(
                            fixed_chunk_ref_path,
                            transform_parameter_object,
                            output_directory=work_dir,
                        ).astype(np.float32)
                        quadrant_chunks.append(
                            QuadrantChunk(
                                core_local_window=core_local_window,
                                chunk_local_window=chunk_local_window,
                                core_source_window=core_source_window,
                                chunk_source_window=chunk_source_window,
                                chunk_transform=chunk_transform,
                                fixed_chunk_window=fixed_chunk_window,
                                fixed_chunk_transform=fixed_chunk_transform,
                                fixed_dx=deformation_field[..., 0],
                                fixed_dy=deformation_field[..., 1],
                            )
                        )

            for b in range(1, moving_src.count + 1):
                if output_on_moving_grid:
                    for chunk in quadrant_chunks:
                        moving_band_data = moving_src.read(b, window=chunk.chunk_source_window).astype(np.float32)
                        moving_band_valid = moving_src.read_masks(b, window=chunk.chunk_source_window).astype(np.float32)
                        if moving_nodata_value is not None:
                            moving_band_valid *= (moving_band_data != moving_nodata_value).astype(np.float32)

                        block_width, block_height = out_dst.block_shapes[b - 1]
                        core_row0 = int(chunk.core_local_window.row_off)
                        core_col0 = int(chunk.core_local_window.col_off)
                        core_row1 = core_row0 + int(chunk.core_local_window.height)
                        core_col1 = core_col0 + int(chunk.core_local_window.width)
                        chunk_row0 = int(chunk.chunk_local_window.row_off)
                        chunk_col0 = int(chunk.chunk_local_window.col_off)

                        for row_off in range(core_row0, core_row1, int(block_height)):
                            for col_off in range(core_col0, core_col1, int(block_width)):
                                win_w = min(int(block_width), core_col1 - col_off)
                                win_h = min(int(block_height), core_row1 - row_off)
                                global_local_block = Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)
                                block_in_chunk = Window(
                                    col_off=int(global_local_block.col_off - chunk_col0),
                                    row_off=int(global_local_block.row_off - chunk_row0),
                                    width=win_w,
                                    height=win_h,
                                )
                                block_transform = rasterio.windows.transform(block_in_chunk, chunk.chunk_transform)
                                x_world, y_world = _pixel_centers_world(block_transform, win_h, win_w)
                                fixed_rows, fixed_cols = _world_to_array_coords(
                                    chunk.fixed_chunk_transform,
                                    x_world,
                                    y_world,
                                )
                                dx_block, dx_valid = _sample_bilinear(
                                    chunk.fixed_dx,
                                    fixed_rows,
                                    fixed_cols,
                                    fill_value=0.0,
                                )
                                dy_block, dy_valid = _sample_bilinear(
                                    chunk.fixed_dy,
                                    fixed_rows,
                                    fixed_cols,
                                    fill_value=0.0,
                                )
                                field_valid = dx_valid & dy_valid
                                source_fixed_rows = fixed_rows + dy_block
                                source_fixed_cols = fixed_cols + dx_block
                                source_x_world, source_y_world = _array_to_world(
                                    chunk.fixed_chunk_transform,
                                    source_fixed_rows,
                                    source_fixed_cols,
                                )
                                source_moving_rows, source_moving_cols = _world_to_array_coords(
                                    chunk.chunk_transform,
                                    source_x_world,
                                    source_y_world,
                                )
                                remapped_block, moving_valid = _sample_bilinear(
                                    moving_band_data,
                                    source_moving_rows,
                                    source_moving_cols,
                                    fill_value=out_nodata,
                                )
                                sampled_mask, mask_valid = _sample_bilinear(
                                    moving_band_valid,
                                    source_moving_rows,
                                    source_moving_cols,
                                    fill_value=0.0,
                                )
                                valid = field_valid & moving_valid & mask_valid & (sampled_mask > 0.0)
                                combined = np.where(valid, remapped_block, out_nodata)
                                out_dst.write(
                                    combined.astype(out_profile["dtype"]),
                                    b,
                                    window=Window(
                                        col_off=int(moving_window.col_off + global_local_block.col_off),
                                        row_off=int(moving_window.row_off + global_local_block.row_off),
                                        width=win_w,
                                        height=win_h,
                                    ),
                                )
                else:
                    moving_band_path = os.path.join(work_dir, f"moving_band_{b:03d}.tif")
                    _stream_reprojected_band_to_tif(
                        moving_band_path,
                        src=moving_src,
                        band_index=b,
                        dst_crs=fixed_src.crs,
                        dst_transform=fixed_src.window_transform(fixed_window),
                        dst_width=int(fixed_window.width),
                        dst_height=int(fixed_window.height),
                        src_nodata=moving_nodata_value,
                        dst_fill_value=out_nodata,
                        output_nodata=moving_nodata_value,
                    )
                    if b == 1:
                        serialized_transform_files = write_transform_parameter_files(
                            transform_parameter_object,
                            os.path.join(work_dir, "transform"),
                        )
                    transformed_band_path = os.path.join(work_dir, f"warped_band_{b:03d}.tif")
                    apply_elastix_transform_subprocess(
                        moving_image_path=moving_band_path,
                        output_image_path=transformed_band_path,
                        parameter_files=serialized_transform_files,
                        reference_image_path=fixed_reg_path,
                        log_to_console=log_to_console,
                    )
                    with rasterio.open(transformed_band_path) as warped_src:
                        for _, block_window in warped_src.block_windows(1):
                            block = warped_src.read(1, window=block_window)
                            out_dst.write(
                                block.astype(out_profile["dtype"]),
                                b,
                                window=block_window,
                            )

    if temp_ctx is not None:
        temp_ctx.cleanup()
        kept_temp = None
    else:
        kept_temp = work_dir

    return AlignmentResult(
        output_image_path=output_image_path,
        temp_dir=kept_temp,
    )
