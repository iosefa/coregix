"""Trim pixels adjacent to exterior invalid regions from an aligned raster."""

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


def _prefix_invalid_lengths(invalid: np.ndarray, axis: int) -> np.ndarray:
    """Return contiguous invalid-run lengths from the low end of an axis."""
    valid = ~invalid
    has_valid = np.any(valid, axis=axis)
    first_valid = np.argmax(valid, axis=axis)
    edge_len = invalid.shape[axis]
    return np.where(has_valid, first_valid, edge_len).astype(np.int64)


def _suffix_invalid_lengths(invalid: np.ndarray, axis: int) -> np.ndarray:
    """Return contiguous invalid-run lengths from the high end of an axis."""
    flipped = np.flip(invalid, axis=axis)
    return _prefix_invalid_lengths(flipped, axis=axis)


def _make_row_trim_mask(invalid: np.ndarray, edge_depth: int) -> np.ndarray:
    rows, cols = invalid.shape
    trim = np.zeros((rows, cols), dtype=bool)

    left_invalid = _prefix_invalid_lengths(invalid, axis=1)
    left_rows = np.where((left_invalid > 0) & (left_invalid < cols))[0]
    for row in left_rows:
        stop = min(cols, int(left_invalid[row]) + edge_depth)
        trim[row, :stop] = True

    right_invalid = _suffix_invalid_lengths(invalid, axis=1)
    right_rows = np.where((right_invalid > 0) & (right_invalid < cols))[0]
    for row in right_rows:
        start = max(0, cols - int(right_invalid[row]) - edge_depth)
        trim[row, start:] = True

    return trim


def _make_col_trim_mask(invalid: np.ndarray, edge_depth: int) -> np.ndarray:
    rows, cols = invalid.shape
    trim = np.zeros((rows, cols), dtype=bool)

    top_invalid = _prefix_invalid_lengths(invalid, axis=0)
    top_cols = np.where((top_invalid > 0) & (top_invalid < rows))[0]
    for col in top_cols:
        stop = min(rows, int(top_invalid[col]) + edge_depth)
        trim[:stop, col] = True

    bottom_invalid = _suffix_invalid_lengths(invalid, axis=0)
    bottom_cols = np.where((bottom_invalid > 0) & (bottom_invalid < rows))[0]
    for col in bottom_cols:
        start = max(0, rows - int(bottom_invalid[col]) - edge_depth)
        trim[start:, col] = True

    return trim


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
        dir=os.path.dirname(input_image_path) or None,
    ) as temp_dir:
        temp_output_path = os.path.join(temp_dir, os.path.basename(final_path))
        shutil.copyfile(input_image_path, temp_output_path)

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

            for row_off in range(0, src.height, row_chunk_size):
                win_h = min(row_chunk_size, src.height - row_off)
                window = Window(0, row_off, src.width, win_h)
                detect = src.read(detection_band, window=window)
                invalid = _invalid_mask(
                    detect,
                    nodata_value=resolved_nodata,
                    invalid_below=invalid_below,
                    invalid_above=invalid_above,
                )
                trim_mask = _make_row_trim_mask(invalid, edge_depth=edge_depth)
                pixels_trimmed += _apply_trim_mask(
                    dst,
                    window=window,
                    trim_mask=trim_mask,
                    nodata_value=resolved_nodata,
                )

            for col_off in range(0, src.width, col_chunk_size):
                win_w = min(col_chunk_size, src.width - col_off)
                window = Window(col_off, 0, win_w, src.height)
                detect = src.read(detection_band, window=window)
                invalid = _invalid_mask(
                    detect,
                    nodata_value=resolved_nodata,
                    invalid_below=invalid_below,
                    invalid_above=invalid_above,
                )
                trim_mask = _make_col_trim_mask(invalid, edge_depth=edge_depth)
                pixels_trimmed += _apply_trim_mask(
                    dst,
                    window=window,
                    trim_mask=trim_mask,
                    nodata_value=resolved_nodata,
                )
        if in_place:
            os.replace(temp_output_path, final_path)
        else:
            shutil.copyfile(temp_output_path, final_path)

    return EdgeTrimResult(
        output_image_path=final_path,
        nodata_value=float(resolved_nodata),
        pixels_trimmed=pixels_trimmed,
    )
