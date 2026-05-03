"""Trim pixels adjacent to invalid regions from an aligned raster."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rasterio
from rasterio.windows import Window


@dataclass
class EdgeTrimResult:
    output_image_path: str
    nodata_value: float
    pixels_trimmed: int


def _invalid_mask(
    data: np.ndarray,
    *,
    nodata_value: Optional[float],
    invalid_below: Optional[float],
    invalid_above: Optional[float],
) -> np.ndarray:
    invalid = np.zeros(data.shape, dtype=bool)
    if nodata_value is not None:
        if np.issubdtype(data.dtype, np.floating):
            invalid |= np.isclose(data, nodata_value)
        else:
            invalid |= data == nodata_value
    if invalid_below is not None:
        invalid |= data <= invalid_below
    if invalid_above is not None:
        invalid |= data >= invalid_above
    return invalid


def _dilate_mask_square(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a mask by ``radius`` pixels using a square footprint."""
    if radius <= 0 or not np.any(mask):
        return mask.copy()

    horizontal = mask.copy()
    for offset in range(1, radius + 1):
        horizontal[:, offset:] |= mask[:, :-offset]
        horizontal[:, :-offset] |= mask[:, offset:]

    dilated = horizontal.copy()
    for offset in range(1, radius + 1):
        dilated[offset:, :] |= horizontal[:-offset, :]
        dilated[:-offset, :] |= horizontal[offset:, :]

    return dilated


def _expand_window(window: Window, padding: int, max_width: int, max_height: int) -> Window:
    col0 = max(0, int(window.col_off) - padding)
    row0 = max(0, int(window.row_off) - padding)
    col1 = min(max_width, int(window.col_off + window.width) + padding)
    row1 = min(max_height, int(window.row_off + window.height) + padding)
    return Window(col_off=col0, row_off=row0, width=col1 - col0, height=row1 - row0)


def _apply_trim_mask(
    dst: rasterio.io.DatasetWriter,
    *,
    window: Window,
    trim_mask: np.ndarray,
    nodata_value: float,
) -> int:
    if not np.any(trim_mask):
        return 0

    ref = dst.read(1, window=window)
    if np.issubdtype(ref.dtype, np.floating):
        newly_trimmed = int(np.logical_and(trim_mask, ~np.isclose(ref, nodata_value)).sum())
    else:
        newly_trimmed = int(np.logical_and(trim_mask, ref != nodata_value).sum())

    for b in range(1, dst.count + 1):
        block = dst.read(b, window=window)
        block[trim_mask] = nodata_value
        dst.write(block, b, window=window)
    return newly_trimmed


def _trim_invalid_edges_windowed(
    src: rasterio.io.DatasetReader,
    dst: rasterio.io.DatasetWriter,
    *,
    detection_band: int,
    edge_depth: int,
    nodata_value: float,
    invalid_below: Optional[float],
    invalid_above: Optional[float],
    row_chunk_size: int,
    col_chunk_size: int,
) -> int:
    pixels_trimmed = 0
    for row_off in range(0, src.height, row_chunk_size):
        win_h = min(row_chunk_size, src.height - row_off)
        for col_off in range(0, src.width, col_chunk_size):
            win_w = min(col_chunk_size, src.width - col_off)
            core_window = Window(col_off, row_off, win_w, win_h)
            read_window = _expand_window(
                core_window,
                edge_depth,
                max_width=src.width,
                max_height=src.height,
            )
            detect = src.read(detection_band, window=read_window)
            invalid = _invalid_mask(
                detect,
                nodata_value=nodata_value,
                invalid_below=invalid_below,
                invalid_above=invalid_above,
            )
            dilated = _dilate_mask_square(invalid, edge_depth)
            core_row0 = int(core_window.row_off - read_window.row_off)
            core_col0 = int(core_window.col_off - read_window.col_off)
            trim_mask = dilated[
                core_row0 : core_row0 + int(core_window.height),
                core_col0 : core_col0 + int(core_window.width),
            ]
            pixels_trimmed += _apply_trim_mask(
                dst,
                window=core_window,
                trim_mask=trim_mask,
                nodata_value=nodata_value,
            )
    return pixels_trimmed


def trim_edge_invalid_pixels(
    input_image_path: str,
    *,
    output_image_path: Optional[str] = None,
    in_place: bool = False,
    edge_depth: int = 8,
    detection_band_index: int = 0,
    invalid_below: Optional[float] = None,
    invalid_above: Optional[float] = None,
    nodata_value: Optional[float] = None,
    row_chunk_size: int = 1024,
    col_chunk_size: int = 1024,
) -> EdgeTrimResult:
    if edge_depth <= 0:
        raise ValueError("edge_depth must be > 0.")
    if detection_band_index < 0:
        raise ValueError("detection_band_index must be >= 0.")
    if row_chunk_size <= 0 or col_chunk_size <= 0:
        raise ValueError("row_chunk_size and col_chunk_size must be > 0.")
    if not os.path.isfile(input_image_path):
        raise FileNotFoundError(input_image_path)
    if in_place and output_image_path is not None:
        raise ValueError("Use either output_image_path or in_place, not both.")
    if not in_place and output_image_path is None:
        raise ValueError("Provide output_image_path or set in_place=True.")

    final_path = input_image_path if in_place else output_image_path
    assert final_path is not None
    os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="edge_trim_",
        dir=os.path.dirname(final_path) or None,
    ) as temp_dir:
        temp_output_path = os.path.join(temp_dir, os.path.basename(final_path))
        shutil.copy2(input_image_path, temp_output_path)

        pixels_trimmed = 0
        with rasterio.open(input_image_path) as src, rasterio.open(temp_output_path, "r+") as dst:
            if detection_band_index >= src.count:
                raise ValueError(
                    f"detection_band_index={detection_band_index} is out of range for raster with {src.count} band(s)."
                )

            resolved_nodata = nodata_value if nodata_value is not None else dst.nodata
            if resolved_nodata is None:
                raise ValueError("Raster has no nodata value; provide nodata_value explicitly.")
            if invalid_below is None and invalid_above is None and src.nodata is None:
                raise ValueError("No invalid criteria available; provide invalid_below and/or invalid_above.")

            if dst.nodata != resolved_nodata:
                dst.nodata = resolved_nodata

            detection_band = detection_band_index + 1

            pixels_trimmed = _trim_invalid_edges_windowed(
                src,
                dst,
                detection_band=detection_band,
                edge_depth=edge_depth,
                nodata_value=resolved_nodata,
                invalid_below=invalid_below,
                invalid_above=invalid_above,
                row_chunk_size=row_chunk_size,
                col_chunk_size=col_chunk_size,
            )

        os.replace(temp_output_path, final_path)

    return EdgeTrimResult(
        output_image_path=final_path,
        nodata_value=float(resolved_nodata),
        pixels_trimmed=pixels_trimmed,
    )
