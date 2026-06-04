"""Apply a saved Coregix transform JSON to a raster."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.windows import Window, from_bounds

from coregix.pipelines.alignment import (
    _make_output_profile,
    _pixel_centers_world,
    _sample_bilinear,
    _to_int_window,
    _world_to_array_coords,
)


@dataclass
class ApplyTransformResult:
    """Summary of applying a saved Coregix transform.

    Attributes:
        output_image_path: Path to the written raster.
        transform_json_path: Transform JSON used for resampling.
        output_width: Output raster width in pixels.
        output_height: Output raster height in pixels.
        output_transform: Output affine transform coefficients in GDAL order.
    """

    output_image_path: str
    transform_json_path: str
    output_width: int
    output_height: int
    output_transform: list[float]


def _affine_from_list(values: list[float]) -> Affine:
    if len(values) != 6:
        raise ValueError("Affine transform metadata must contain 6 values.")
    return Affine(*[float(value) for value in values])


def _window_from_dict(values: dict[str, Any]) -> Window:
    return Window(
        col_off=float(values["col_off"]),
        row_off=float(values["row_off"]),
        width=float(values["width"]),
        height=float(values["height"]),
    )


def _matrix_from_metadata(metadata: dict[str, Any]) -> np.ndarray:
    try:
        matrix = np.asarray(metadata["target_to_source"]["matrix"], dtype=np.float64)
    except KeyError as exc:
        raise ValueError("Transform JSON is missing target_to_source.matrix.") from exc
    if matrix.shape != (3, 3):
        raise ValueError("target_to_source.matrix must be a 3x3 homogeneous matrix.")
    return matrix


def _transform_xy(matrix: np.ndarray, x_world: np.ndarray, y_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_x = matrix[0, 0] * x_world + matrix[0, 1] * y_world + matrix[0, 2]
    source_y = matrix[1, 0] * x_world + matrix[1, 1] * y_world + matrix[1, 2]
    return source_x, source_y


def _transform_points(matrix: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack(
        [points_xy[:, 0], points_xy[:, 1], np.ones(points_xy.shape[0], dtype=np.float64)]
    )
    return (homogeneous @ matrix.T)[:, :2]


def _window_corners_world(window: Window, transform: Affine) -> np.ndarray:
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


def _expand_window(window: Window, padding: int, max_width: int, max_height: int) -> Window:
    return _to_int_window(
        Window(
            col_off=window.col_off - padding,
            row_off=window.row_off - padding,
            width=window.width + 2 * padding,
            height=window.height + 2 * padding,
        ),
        max_width=max_width,
        max_height=max_height,
    )


def _source_window_for_target_window(
    *,
    target_window: Window,
    target_transform: Affine,
    source_transform: Affine,
    target_to_source_matrix: np.ndarray,
    source_width: int,
    source_height: int,
    padding_pixels: int = 2,
) -> Window:
    corners = _window_corners_world(target_window, target_transform)
    source_corners = _transform_points(target_to_source_matrix, corners)
    source_window = _to_int_window(
        from_bounds(
            left=float(source_corners[:, 0].min()),
            bottom=float(source_corners[:, 1].min()),
            right=float(source_corners[:, 0].max()),
            top=float(source_corners[:, 1].max()),
            transform=source_transform,
        ),
        max_width=source_width,
        max_height=source_height,
    )
    return _expand_window(source_window, padding_pixels, source_width, source_height)


def _metadata_crs(value: Any) -> Optional[CRS]:
    if value is None:
        return None
    return CRS.from_string(str(value))


def _validate_moving_raster(
    moving_src: rasterio.DatasetReader,
    metadata: dict[str, Any],
    moving_image_path: str,
    *,
    allow_mismatch: bool,
) -> None:
    paths = metadata.get("paths", {})
    json_moving_path = paths.get("moving_image")
    if json_moving_path and not allow_mismatch:
        if os.path.abspath(str(json_moving_path)) != os.path.abspath(moving_image_path):
            raise ValueError(
                "Moving image path does not match transform JSON. "
                "Use allow_moving_image_mismatch=True only if the raster is an equivalent copy."
            )

    raster_meta = metadata.get("rasters", {}).get("moving", {})
    expected_width = raster_meta.get("width")
    expected_height = raster_meta.get("height")
    if expected_width is not None and int(expected_width) != moving_src.width and not allow_mismatch:
        raise ValueError("Moving raster width does not match transform JSON.")
    if expected_height is not None and int(expected_height) != moving_src.height and not allow_mismatch:
        raise ValueError("Moving raster height does not match transform JSON.")

    expected_transform_values = raster_meta.get("transform")
    if expected_transform_values is not None and not allow_mismatch:
        expected_transform = _affine_from_list(expected_transform_values)
        if not np.allclose(tuple(expected_transform)[:6], tuple(moving_src.transform)[:6]):
            raise ValueError("Moving raster transform does not match transform JSON.")

    expected_crs = _metadata_crs(metadata.get("crs", {}).get("moving"))
    if expected_crs is not None and moving_src.crs is not None and not allow_mismatch:
        if expected_crs != moving_src.crs:
            raise ValueError("Moving raster CRS does not match transform JSON.")


def _copy_source_metadata(src: rasterio.DatasetReader, dst: rasterio.io.DatasetWriter) -> None:
    try:
        dst.colorinterp = src.colorinterp
    except Exception:
        pass
    try:
        dst.scales = src.scales
        dst.offsets = src.offsets
    except Exception:
        pass
    for band_index in range(1, src.count + 1):
        desc = src.descriptions[band_index - 1]
        if desc:
            dst.set_band_description(band_index, desc)
        band_tags = src.tags(band_index).copy()
        for key in list(band_tags.keys()):
            if key.upper().startswith("STATISTICS_"):
                band_tags.pop(key, None)
        dst.update_tags(band_index, **band_tags)
        dst.update_tags(
            band_index,
            STATISTICS_MINIMUM="",
            STATISTICS_MAXIMUM="",
            STATISTICS_MEAN="",
            STATISTICS_STDDEV="",
        )


def _initialize_output(
    *,
    moving_src: rasterio.DatasetReader,
    out_dst: rasterio.io.DatasetWriter,
    output_on_moving_grid: bool,
    output_nodata: float,
) -> None:
    if output_on_moving_grid:
        for band_index in range(1, moving_src.count + 1):
            for _, block_window in out_dst.block_windows(band_index):
                src_block = moving_src.read(band_index, window=block_window)
                out_dst.write(src_block.astype(out_dst.dtypes[band_index - 1]), band_index, window=block_window)
    else:
        for band_index in range(1, moving_src.count + 1):
            dtype = out_dst.dtypes[band_index - 1]
            for _, block_window in out_dst.block_windows(band_index):
                fill = np.full(
                    (int(block_window.height), int(block_window.width)),
                    output_nodata,
                    dtype=dtype,
                )
                out_dst.write(fill, band_index, window=block_window)


def _iter_target_blocks(
    *,
    out_dst: rasterio.io.DatasetWriter,
    target_window: Window,
    block_shape: tuple[int, int],
) -> list[Window]:
    block_height, block_width = block_shape
    row0 = int(target_window.row_off)
    col0 = int(target_window.col_off)
    row1 = row0 + int(target_window.height)
    col1 = col0 + int(target_window.width)
    windows: list[Window] = []
    for row_off in range(row0, row1, int(block_height)):
        for col_off in range(col0, col1, int(block_width)):
            win_w = min(int(block_width), col1 - col_off)
            win_h = min(int(block_height), row1 - row_off)
            if win_w <= 0 or win_h <= 0:
                continue
            window = Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)
            if window.col_off >= out_dst.width or window.row_off >= out_dst.height:
                continue
            clipped = Window(
                col_off=window.col_off,
                row_off=window.row_off,
                width=min(window.width, out_dst.width - int(window.col_off)),
                height=min(window.height, out_dst.height - int(window.row_off)),
            )
            if clipped.width > 0 and clipped.height > 0:
                windows.append(clipped)
    return windows


def _write_transformed_blocks(
    *,
    moving_src: rasterio.DatasetReader,
    out_dst: rasterio.io.DatasetWriter,
    target_window: Window,
    target_transform: Affine,
    target_to_source_matrix: np.ndarray,
    output_nodata: float,
) -> None:
    for band_index in range(1, moving_src.count + 1):
        for block_window in _iter_target_blocks(
            out_dst=out_dst,
            target_window=target_window,
            block_shape=out_dst.block_shapes[band_index - 1],
        ):
            source_window = _source_window_for_target_window(
                target_window=block_window,
                target_transform=target_transform,
                source_transform=moving_src.transform,
                target_to_source_matrix=target_to_source_matrix,
                source_width=moving_src.width,
                source_height=moving_src.height,
            )
            if source_window.width <= 0 or source_window.height <= 0:
                fill = np.full(
                    (int(block_window.height), int(block_window.width)),
                    output_nodata,
                    dtype=out_dst.dtypes[band_index - 1],
                )
                out_dst.write(fill, band_index, window=block_window)
                continue

            moving_band_data = moving_src.read(band_index, window=source_window).astype(np.float32)
            moving_valid_data = moving_src.read_masks(band_index, window=source_window).astype(np.float32)
            if moving_src.nodata is not None:
                moving_valid_data *= (moving_band_data != moving_src.nodata).astype(np.float32)

            block_transform = rasterio.windows.transform(block_window, target_transform)
            x_world, y_world = _pixel_centers_world(
                block_transform,
                int(block_window.height),
                int(block_window.width),
            )
            source_x_world, source_y_world = _transform_xy(target_to_source_matrix, x_world, y_world)
            source_rows, source_cols = _world_to_array_coords(
                moving_src.window_transform(source_window),
                source_x_world,
                source_y_world,
            )
            remapped_block, moving_sample_valid = _sample_bilinear(
                moving_band_data,
                source_rows,
                source_cols,
                fill_value=output_nodata,
            )
            sampled_mask, mask_valid = _sample_bilinear(
                moving_valid_data,
                source_rows,
                source_cols,
                fill_value=0.0,
            )
            valid = moving_sample_valid & mask_valid & (sampled_mask > 0.0)
            combined = np.where(valid, remapped_block, output_nodata)
            out_dst.write(combined.astype(out_dst.dtypes[band_index - 1]), band_index, window=block_window)


def apply_coregix_transform(
    *,
    moving_image_path: str,
    transform_json_path: str,
    output_image_path: str,
    allow_moving_image_mismatch: bool = False,
    trim_edge_invalid: bool = False,
    edge_trim_depth: int = 8,
    edge_trim_detection_band_index: int = 0,
    edge_trim_invalid_below: Optional[float] = None,
    edge_trim_invalid_above: Optional[float] = None,
) -> ApplyTransformResult:
    """Apply a saved Coregix transform JSON to a source raster.

    The transform JSON must contain ``target_to_source.matrix`` and output grid
    metadata produced by ``align_image_pair(..., output_transform_json_path=...)``.
    The registration solve is not rerun; this function only resamples the source
    raster onto the saved output grid.
    """
    if not os.path.isfile(moving_image_path):
        raise FileNotFoundError(moving_image_path)
    if not os.path.isfile(transform_json_path):
        raise FileNotFoundError(transform_json_path)
    if edge_trim_depth <= 0:
        raise ValueError("edge_trim_depth must be > 0.")
    if edge_trim_detection_band_index < 0:
        raise ValueError("edge_trim_detection_band_index must be >= 0.")

    metadata = json.loads(Path(transform_json_path).read_text(encoding="utf-8"))
    target_to_source_matrix = _matrix_from_metadata(metadata)
    output_meta = metadata.get("rasters", {}).get("output")
    if not output_meta:
        raise ValueError("Transform JSON is missing rasters.output metadata.")
    output_width = int(output_meta["width"])
    output_height = int(output_meta["height"])
    output_transform = _affine_from_list(output_meta["transform"])
    if output_width <= 0 or output_height <= 0:
        raise ValueError("Output width and height in transform JSON must be positive.")

    options = metadata.get("options", {})
    output_on_moving_grid = bool(options.get("output_on_moving_grid", True))

    output_nodata_value = options.get("output_nodata")
    if output_nodata_value is None:
        output_nodata_value = 0.0
    output_nodata = float(output_nodata_value)

    os.makedirs(os.path.dirname(output_image_path) or ".", exist_ok=True)

    final_write_path = output_image_path
    temp_dir_ctx: Optional[tempfile.TemporaryDirectory[str]] = None
    if trim_edge_invalid:
        temp_dir_ctx = tempfile.TemporaryDirectory(
            prefix="coregix_apply_",
            dir=os.path.dirname(output_image_path) or None,
        )
        final_write_path = os.path.join(temp_dir_ctx.name, os.path.basename(output_image_path))

    try:
        with rasterio.open(moving_image_path) as moving_src:
            _validate_moving_raster(
                moving_src,
                metadata,
                moving_image_path,
                allow_mismatch=allow_moving_image_mismatch,
            )
            fixed_crs = _metadata_crs(metadata.get("crs", {}).get("fixed"))
            output_crs = moving_src.crs if output_on_moving_grid else fixed_crs or moving_src.crs
            out_profile = _make_output_profile(
                moving_src.profile,
                count=moving_src.count,
                dtype=moving_src.dtypes[0],
                nodata=output_nodata,
                width=output_width,
                height=output_height,
                transform=output_transform,
            )
            out_profile["crs"] = output_crs

            if output_on_moving_grid:
                moving_window_meta = metadata.get("windows", {}).get("moving")
                target_window = (
                    _window_from_dict(moving_window_meta)
                    if moving_window_meta is not None
                    else Window(0, 0, output_width, output_height)
                )
            else:
                target_window = Window(0, 0, output_width, output_height)

            with rasterio.open(final_write_path, "w+", **out_profile) as out_dst:
                _copy_source_metadata(moving_src, out_dst)
                _initialize_output(
                    moving_src=moving_src,
                    out_dst=out_dst,
                    output_on_moving_grid=output_on_moving_grid,
                    output_nodata=output_nodata,
                )
                _write_transformed_blocks(
                    moving_src=moving_src,
                    out_dst=out_dst,
                    target_window=target_window,
                    target_transform=output_transform,
                    target_to_source_matrix=target_to_source_matrix,
                    output_nodata=output_nodata,
                )

        if trim_edge_invalid:
            from coregix.postprocess import trim_edge_invalid_pixels

            trim_edge_invalid_pixels(
                input_image_path=final_write_path,
                output_image_path=output_image_path,
                edge_depth=edge_trim_depth,
                detection_band_index=edge_trim_detection_band_index,
                invalid_below=edge_trim_invalid_below,
                invalid_above=edge_trim_invalid_above,
                nodata_value=output_nodata,
            )
    finally:
        if temp_dir_ctx is not None:
            temp_dir_ctx.cleanup()

    return ApplyTransformResult(
        output_image_path=output_image_path,
        transform_json_path=transform_json_path,
        output_width=output_width,
        output_height=output_height,
        output_transform=[float(value) for value in output_transform[:6]],
    )


__all__ = ["ApplyTransformResult", "apply_coregix_transform"]
