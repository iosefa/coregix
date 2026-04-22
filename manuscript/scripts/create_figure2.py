#!/usr/bin/env python3
"""Create Figure 2 comparing no alignment versus gradient-based alignment."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from html import escape

import numpy as np
from osgeo import gdal, ogr
import rasterio
from affine import Affine
from rasterio.features import rasterize
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from rasterio.transform import rowcol
from rasterio.warp import reproject
from rasterio.windows import Window

from coregix.pipelines.alignment import (
    DEFAULT_ALIGNMENT_PARAMETER_MAPS,
    _edge_proxy,
    _resolve_nodata,
    _write_single_band_tif,
)
from coregix.preprocess.registration import apply_elastix_transform_array, estimate_elastix_transform
from create_figure1 import _reproject_to_fixed_grid as _reproject_band_to_fixed_grid


DEFAULT_MOVING_IMAGE = "examples/data/moving/img_sat_lg.tif"
DEFAULT_FIXED_IMAGE = "examples/data/fixed/intensity_lg.tif"
DEFAULT_GCP_IMAGE = "examples/data/aligned/gcp_lg.tif"
DEFAULT_GCP_POINTS = "manuscript/data/gcp_img.points"
DEFAULT_GCP_GEOJSON = "/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/20200107_31cm_WV03_BAB_050340850010/OrthoFromUSGSLidar/20JAN07211752-M1BS-050340850010_01_P001_points.geojson"
DEFAULT_FULL_MOVING_IMAGE = "/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/20200107_31cm_WV03_BAB_050340850010/Mul_FLAASH_OrthoFromDefaultRPC_Pansharp/20JAN07211752-M1BS-050340850010_01_P001_FLAASH_OrthoFromDefaultRPC_Pansharps.tif"
DEFAULT_ORTHORITY_DEM = "/mnt/x/Imagery/Elevation/DEM/BigIsland/Entirety/SRTM/Hawaii_SRTM_GL1Ellip/Hawaii_SRTM_GL1Ellip.tif"
DEFAULT_MOVING_CROWNS = "manuscript/data/crowns_moving.gpkg"
DEFAULT_FIXED_CROWNS = "manuscript/data/crowns_fixed.gpkg"
DEFAULT_OUTPUT_SVG = "manuscript/figures/figure2_registration_comparison.svg"
DEFAULT_OUTPUT_PREVIEW = "manuscript/figures/figure2_registration_comparison_preview.png"
DEFAULT_OUTPUT_SVG_WITH_LEGEND = "manuscript/figures/figure2_registration_comparison_with_legend.svg"
DEFAULT_OUTPUT_PREVIEW_WITH_LEGEND = "manuscript/figures/figure2_registration_comparison_with_legend_preview.png"


@dataclass
class OverlayPanel:
    label: str
    title: str
    image: np.ndarray
    metric_text: str | None = None


_BITMAP_FONT = {
    " ": ["000", "000", "000", "000", "000", "000", "000"],
    "-": ["000", "000", "000", "111", "000", "000", "000"],
    ".": ["000", "000", "000", "000", "000", "010", "010"],
    ":": ["000", "010", "010", "000", "010", "010", "000"],
    "0": ["111", "101", "101", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "010", "010", "111"],
    "2": ["111", "001", "001", "111", "100", "100", "111"],
    "3": ["111", "001", "001", "111", "001", "001", "111"],
    "4": ["101", "101", "101", "111", "001", "001", "001"],
    "5": ["111", "100", "100", "111", "001", "001", "111"],
    "6": ["111", "100", "100", "111", "101", "101", "111"],
    "7": ["111", "001", "001", "001", "001", "001", "001"],
    "8": ["111", "101", "101", "111", "101", "101", "111"],
    "9": ["111", "101", "101", "111", "001", "001", "111"],
    "A": ["010", "101", "101", "111", "101", "101", "101"],
    "B": ["110", "101", "101", "110", "101", "101", "110"],
    "C": ["011", "100", "100", "100", "100", "100", "011"],
    "D": ["110", "101", "101", "101", "101", "101", "110"],
    "E": ["111", "100", "100", "110", "100", "100", "111"],
    "G": ["011", "100", "100", "101", "101", "101", "011"],
    "I": ["111", "010", "010", "010", "010", "010", "111"],
    "L": ["100", "100", "100", "100", "100", "100", "111"],
    "M": ["101", "111", "111", "101", "101", "101", "101"],
    "N": ["101", "111", "111", "111", "111", "111", "101"],
    "O": ["111", "101", "101", "101", "101", "101", "111"],
    "P": ["110", "101", "101", "110", "100", "100", "100"],
    "R": ["110", "101", "101", "110", "101", "101", "101"],
    "S": ["011", "100", "100", "010", "001", "001", "110"],
    "T": ["111", "010", "010", "010", "010", "010", "010"],
    "X": ["101", "101", "101", "010", "101", "101", "101"],
    "Y": ["101", "101", "101", "010", "010", "010", "010"],
    "m": ["000", "000", "110", "111", "101", "101", "101"],
}


def _draw_rect(canvas: np.ndarray, *, x: int, y: int, width: int, height: int, color: tuple[int, int, int]) -> None:
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(canvas.shape[2], x + width)
    y1 = min(canvas.shape[1], y + height)
    if x1 <= x0 or y1 <= y0:
        return
    canvas[:, y0:y1, x0:x1] = np.asarray(color, dtype=np.uint8)[:, None, None]


def _draw_bitmap_text(
    canvas: np.ndarray,
    *,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int] = (17, 17, 17),
    scale: int = 2,
    spacing: int = 1,
) -> None:
    cursor_x = x
    for ch in text:
        glyph = _BITMAP_FONT.get(ch, _BITMAP_FONT[" "])
        glyph_w = len(glyph[0])
        glyph_h = len(glyph)
        for row in range(glyph_h):
            for col in range(glyph_w):
                if glyph[row][col] == "1":
                    _draw_rect(
                        canvas,
                        x=cursor_x + col * scale,
                        y=y + row * scale,
                        width=scale,
                        height=scale,
                        color=color,
                    )
        cursor_x += glyph_w * scale + spacing


def _bitmap_text_width(text: str, *, scale: int = 2, spacing: int = 1) -> int:
    width = 0
    for idx, ch in enumerate(text):
        glyph = _BITMAP_FONT.get(ch, _BITMAP_FONT[" "])
        width += len(glyph[0]) * scale
        if idx < len(text) - 1:
            width += spacing
    return width


def _resolve_nodata(src: rasterio.DatasetReader) -> float | None:
    return src.nodata


def _valid_mask(src: rasterio.DatasetReader, band_index_1based: int, data: np.ndarray) -> np.ndarray:
    mask = src.read_masks(band_index_1based) > 0
    nodata = _resolve_nodata(src)
    if nodata is not None:
        mask &= data != nodata
    return mask


def _percentile_scale(data: np.ndarray, valid_mask: np.ndarray, *, low: float, high: float) -> np.ndarray:
    out = np.zeros(data.shape, dtype=np.uint8)
    if not np.any(valid_mask):
        return out

    values = data[valid_mask]
    lo = float(np.percentile(values, low))
    hi = float(np.percentile(values, high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        scaled = np.zeros_like(data, dtype=np.float32)
    else:
        scaled = (data.astype(np.float32) - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    out[valid_mask] = np.round(255.0 * scaled[valid_mask]).astype(np.uint8)
    return out


def _edge_mask(edge_data: np.ndarray, valid_mask: np.ndarray, *, percentile: float = 94.0) -> np.ndarray:
    mask = np.zeros(edge_data.shape, dtype=bool)
    if not np.any(valid_mask):
        return mask
    threshold = float(np.percentile(edge_data[valid_mask], percentile))
    mask = valid_mask & (edge_data >= threshold)
    for row_shift, col_shift in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = np.zeros_like(mask)
        src_rows = slice(max(0, -row_shift), mask.shape[0] - max(0, row_shift))
        dst_rows = slice(max(0, row_shift), mask.shape[0] - max(0, -row_shift))
        src_cols = slice(max(0, -col_shift), mask.shape[1] - max(0, col_shift))
        dst_cols = slice(max(0, col_shift), mask.shape[1] - max(0, -col_shift))
        shifted[dst_rows, dst_cols] = mask[src_rows, src_cols]
        mask |= shifted
    return mask


def _shift_bool(mask: np.ndarray, row_shift: int, col_shift: int) -> np.ndarray:
    shifted = np.zeros_like(mask, dtype=bool)
    src_rows = slice(max(0, -row_shift), mask.shape[0] - max(0, row_shift))
    dst_rows = slice(max(0, row_shift), mask.shape[0] - max(0, -row_shift))
    src_cols = slice(max(0, -col_shift), mask.shape[1] - max(0, col_shift))
    dst_cols = slice(max(0, col_shift), mask.shape[1] - max(0, -col_shift))
    shifted[dst_rows, dst_cols] = mask[src_rows, src_cols]
    return shifted


def _binary_dilation(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.astype(bool).copy()
    for _ in range(iterations):
        expanded = out.copy()
        for row_shift in (-1, 0, 1):
            for col_shift in (-1, 0, 1):
                expanded |= _shift_bool(out, row_shift, col_shift)
        out = expanded
    return out


def _binary_erosion(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.astype(bool).copy()
    for _ in range(iterations):
        eroded = out.copy()
        for row_shift in (-1, 0, 1):
            for col_shift in (-1, 0, 1):
                eroded &= _shift_bool(out, row_shift, col_shift)
        out = eroded
    return out


def _binary_opening(mask: np.ndarray) -> np.ndarray:
    return _binary_dilation(_binary_erosion(mask, iterations=1), iterations=1)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    background = ~mask
    reachable = np.zeros_like(mask, dtype=bool)
    reachable[0, :] = background[0, :]
    reachable[-1, :] = background[-1, :]
    reachable[:, 0] |= background[:, 0]
    reachable[:, -1] |= background[:, -1]

    while True:
        expanded = _binary_dilation(reachable, iterations=1) & background
        if np.array_equal(expanded, reachable):
            break
        reachable = expanded
    return ~reachable


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    mask = mask.astype(bool)
    labels = np.zeros(mask.shape, dtype=np.int32)
    current = 0
    rows, cols = mask.shape
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            current += 1
            stack = [(r, c)]
            labels[r, c] = current
            while stack:
                rr, cc = stack.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = rr + dr
                        nc = cc + dc
                        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                            continue
                        if mask[nr, nc] and labels[nr, nc] == 0:
                            labels[nr, nc] = current
                            stack.append((nr, nc))
    return labels, current


def _center_of_mass(mask: np.ndarray) -> tuple[float, float]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return float(mask.shape[0] / 2), float(mask.shape[1] / 2)
    center = coords.mean(axis=0)
    return float(center[0]), float(center[1])


def _integral_image(data: np.ndarray) -> np.ndarray:
    padded = np.pad(data, ((1, 0), (1, 0)), mode="constant")
    return padded.cumsum(axis=0).cumsum(axis=1)


def _window_sum(integral: np.ndarray, top: int, left: int, height: int, width: int) -> float:
    bottom = top + height
    right = left + width
    return float(
        integral[bottom, right]
        - integral[top, right]
        - integral[bottom, left]
        + integral[top, left]
    )


def _choose_improvement_crop(
    fixed_edge: np.ndarray,
    unaligned_edge: np.ndarray,
    raw_edge: np.ndarray,
    aligned_edge: np.ndarray,
    mutual_valid: np.ndarray,
    *,
    crop_size: int,
) -> Window:
    height, width = mutual_valid.shape
    crop_h = min(crop_size, height)
    crop_w = min(crop_size, width)
    if crop_h == height and crop_w == width:
        return Window(0, 0, width, height)

    unaligned_error = np.abs(unaligned_edge - fixed_edge)
    raw_error = np.abs(raw_edge - fixed_edge)
    aligned_error = np.abs(aligned_edge - fixed_edge)
    improvement = np.clip(np.minimum(unaligned_error, raw_error) - aligned_error, 0.0, None)
    score_image = np.where(mutual_valid, improvement + 0.15 * fixed_edge, 0.0).astype(np.float64)
    valid_counts = mutual_valid.astype(np.float64)
    score_integral = _integral_image(score_image)
    valid_integral = _integral_image(valid_counts)

    best_score = -np.inf
    best = Window(0, 0, crop_w, crop_h)
    min_valid_fraction = 0.95
    min_valid_pixels = crop_h * crop_w * min_valid_fraction
    for row_off in range(0, height - crop_h + 1):
        for col_off in range(0, width - crop_w + 1):
            valid_pixels = _window_sum(valid_integral, row_off, col_off, crop_h, crop_w)
            if valid_pixels < min_valid_pixels:
                continue
            score = _window_sum(score_integral, row_off, col_off, crop_h, crop_w)
            if score > best_score:
                best_score = score
                best = Window(col_off, row_off, crop_w, crop_h)
    return best


def _crop(arr: np.ndarray, window: Window) -> np.ndarray:
    row0 = int(window.row_off)
    row1 = row0 + int(window.height)
    col0 = int(window.col_off)
    col1 = col0 + int(window.width)
    if arr.ndim == 2:
        return arr[row0:row1, col0:col1]
    if arr.ndim == 3:
        return arr[:, row0:row1, col0:col1]
    raise ValueError(f"Unsupported array rank for crop: {arr.ndim}")


def _resize_band(data: np.ndarray, out_height: int, out_width: int) -> np.ndarray:
    dst = np.zeros((out_height, out_width), dtype=np.float32)
    reproject(
        source=data.astype(np.float32),
        destination=dst,
        src_transform=Affine.identity(),
        src_crs="EPSG:3857",
        dst_transform=Affine.scale(data.shape[1] / out_width, data.shape[0] / out_height),
        dst_crs="EPSG:3857",
        resampling=Resampling.bilinear,
    )
    return dst


def _resize_rgb(rgb: np.ndarray, out_height: int, out_width: int) -> np.ndarray:
    resized = np.zeros((3, out_height, out_width), dtype=np.uint8)
    for idx in range(3):
        band = _resize_band(rgb[idx], out_height, out_width)
        resized[idx] = np.clip(np.round(band), 0, 255).astype(np.uint8)
    return resized


def _component_mask_from_edge(edge: np.ndarray, valid_mask: np.ndarray, *, percentile: float = 94.0) -> np.ndarray:
    strong = _edge_mask(edge, valid_mask, percentile=percentile)
    mask = _binary_dilation(strong, iterations=2)
    mask = _fill_holes(mask)
    mask = _binary_opening(mask)
    mask &= valid_mask
    return mask


def _select_component(mask: np.ndarray, target_rc: tuple[int, int], *, min_area: int = 40) -> np.ndarray:
    labels, count = _label_components(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)

    target_r, target_c = target_rc
    if 0 <= target_r < labels.shape[0] and 0 <= target_c < labels.shape[1]:
        label_id = int(labels[target_r, target_c])
        if label_id > 0 and int((labels == label_id).sum()) >= min_area:
            return labels == label_id

    best_label = 0
    best_distance = np.inf
    for label_id in range(1, count + 1):
        component = labels == label_id
        area = int(component.sum())
        if area < min_area:
            continue
        center_r, center_c = _center_of_mass(component)
        distance = float((center_r - target_r) ** 2 + (center_c - target_c) ** 2)
        if distance < best_distance:
            best_distance = distance
            best_label = label_id
    return labels == best_label if best_label else np.zeros_like(mask, dtype=bool)


def _pick_best_fixed_component(component_mask: np.ndarray, score_image: np.ndarray) -> np.ndarray:
    labels, count = _label_components(component_mask)
    if count == 0:
        return np.zeros_like(component_mask, dtype=bool)

    rows, cols = component_mask.shape
    center_r = (rows - 1) / 2.0
    center_c = (cols - 1) / 2.0
    max_area = max(120, (rows * cols) // 8)
    best_label = 0
    best_score = -np.inf

    for label_id in range(1, count + 1):
        component = labels == label_id
        area = int(component.sum())
        if area < 60 or area > max_area:
            continue

        coords = np.argwhere(component)
        row0, col0 = coords.min(axis=0)
        row1, col1 = coords.max(axis=0) + 1
        margin = 12
        er0 = max(0, row0 - margin)
        ec0 = max(0, col0 - margin)
        er1 = min(rows, row1 + margin)
        ec1 = min(cols, col1 + margin)
        neighborhood = labels[er0:er1, ec0:ec1]
        neighbor_pixels = int(((neighborhood > 0) & (neighborhood != label_id)).sum())
        isolation = 1.0 / (1.0 + neighbor_pixels / max(area, 1))

        comp_r, comp_c = _center_of_mass(component)
        center_dist = np.hypot((comp_r - center_r) / max(rows, 1), (comp_c - center_c) / max(cols, 1))

        improvement = float(score_image[component].mean()) if np.any(component) else 0.0
        score = improvement * isolation - 0.35 * center_dist
        if score > best_score:
            best_score = score
            best_label = label_id

    return labels == best_label if best_label else np.zeros_like(component_mask, dtype=bool)


def _load_guide_geometry(path: str) -> tuple[tuple[float, float] | None, tuple[float, float, float, float] | None]:
    if not path or not os.path.exists(path):
        return None, None
    ds = ogr.Open(path)
    if ds is None:
        return None, None

    point_xy = None
    crop_bounds = None

    layer = ds.GetLayerByName("figure2_selected_crown")
    if layer is not None and layer.GetFeatureCount() > 0:
        feat = layer.GetNextFeature()
        geom = feat.GetGeometryRef()
        if geom is not None and geom.GetGeometryName().upper() == "POINT":
            point_xy = (geom.GetX(), geom.GetY())

    layer = ds.GetLayerByName("figure2_selected_crown_bbox")
    if layer is not None and layer.GetFeatureCount() > 0:
        feat = layer.GetNextFeature()
        geom = feat.GetGeometryRef()
        if geom is not None:
            env = geom.GetEnvelope()
            crop_bounds = (env[0], env[2], env[1], env[3])

    ds = None
    return point_xy, crop_bounds


def _window_from_bounds(
    bounds_xy: tuple[float, float, float, float],
    transform,
    width: int,
    height: int,
    *,
    padding_pixels: int = 28,
) -> Window:
    left, bottom, right, top = bounds_xy
    row_top, col_left = rowcol(transform, left, top)
    row_bottom, col_right = rowcol(transform, right, bottom)
    row0 = min(row_top, row_bottom)
    row1 = max(row_top, row_bottom) + 1
    col0 = min(col_left, col_right)
    col1 = max(col_left, col_right) + 1

    row0 -= padding_pixels
    row1 += padding_pixels
    col0 -= padding_pixels
    col1 += padding_pixels

    crop_h = max(1, row1 - row0)
    crop_w = max(1, col1 - col0)
    size = max(crop_h, crop_w)
    center_r = (row0 + row1) / 2.0
    center_c = (col0 + col1) / 2.0
    row0 = int(round(center_r - size / 2.0))
    col0 = int(round(center_c - size / 2.0))
    row0 = max(0, min(row0, height - size))
    col0 = max(0, min(col0, width - size))
    size = min(size, height - row0, width - col0)
    return Window(col0, row0, size, size)


def _boundary(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    eroded = _binary_erosion(mask, iterations=1)
    boundary = mask & ~eroded
    return _binary_dilation(boundary, iterations=1)


def _mask_dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    denom = int(a.sum() + b.sum())
    return float((2.0 * inter / denom) if denom else 0.0)


def _symmetric_boundary_rmse(a: np.ndarray, b: np.ndarray) -> float:
    a_pts = np.argwhere(_boundary(a))
    b_pts = np.argwhere(_boundary(b))
    if a_pts.size == 0 or b_pts.size == 0:
        return float("inf")
    dists = np.sqrt(((a_pts[:, None, :] - b_pts[None, :, :]) ** 2).sum(axis=2))
    a_to_b = dists.min(axis=1)
    b_to_a = dists.min(axis=0)
    return float(np.sqrt((np.mean(a_to_b**2) + np.mean(b_to_a**2)) / 2.0))


def _overlay_crown_panel(
    background_gray: np.ndarray,
    valid_mask: np.ndarray,
    fixed_component: np.ndarray,
    moving_component: np.ndarray,
) -> np.ndarray:
    background = _percentile_scale(background_gray, valid_mask, low=2, high=98).astype(np.float32)
    base = np.where(valid_mask, 18.0 + 0.82 * background, 245.0).astype(np.float32)
    rgb = np.stack([base, base, base], axis=0)
    fixed_boundary = _boundary(fixed_component)
    moving_boundary = _boundary(moving_component)
    overlap_boundary = fixed_boundary & moving_boundary
    fixed_only = fixed_boundary & ~moving_boundary
    moving_only = moving_boundary & ~fixed_boundary
    rgb[:, fixed_only] = np.array([235.0, 135.0, 30.0], dtype=np.float32)[:, None]
    rgb[:, moving_only] = np.array([15.0, 190.0, 235.0], dtype=np.float32)[:, None]
    rgb[:, overlap_boundary] = np.array([255.0, 255.0, 120.0], dtype=np.float32)[:, None]
    return rgb.astype(np.uint8)


def _png_bytes(rgb: np.ndarray) -> bytes:
    profile = {
        "driver": "PNG",
        "width": int(rgb.shape[2]),
        "height": int(rgb.shape[1]),
        "count": 3,
        "dtype": "uint8",
    }
    with MemoryFile() as memfile:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with memfile.open(**profile) as dst:
                dst.write(rgb)
        return memfile.read()


def _write_png(path: str, rgb: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    profile = {
        "driver": "PNG",
        "width": int(rgb.shape[2]),
        "height": int(rgb.shape[1]),
        "count": 3,
        "dtype": "uint8",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(rgb)


def _load_single_polygon(path: str) -> tuple[dict, tuple[float, float, float, float]]:
    ogr.DontUseExceptions()
    ds = ogr.Open(path)
    if ds is None:
        raise RuntimeError(f"Failed to open vector file: {path}")
    layer = ds.GetLayer(0)
    if layer is None or layer.GetFeatureCount() < 1:
        raise RuntimeError(f"No polygon features found in {path}")
    feat = layer.GetNextFeature()
    geom = feat.GetGeometryRef()
    if geom is None:
        raise RuntimeError(f"Feature in {path} has no geometry")
    geojson = json.loads(geom.ExportToJson())
    env = geom.GetEnvelope()
    ds = None
    return geojson, (env[0], env[2], env[1], env[3])


def _transform_coords(coords, affine_transform: Affine):
    if not coords:
        return coords
    first = coords[0]
    if isinstance(first, (float, int)):
        x, y = coords[:2]
        tx, ty = affine_transform * (x, y)
        if len(coords) > 2:
            return [tx, ty, *coords[2:]]
        return [tx, ty]
    return [_transform_coords(part, affine_transform) for part in coords]


def _transform_geojson_geometry(geometry: dict, affine_transform: Affine) -> dict:
    transformed = dict(geometry)
    transformed["coordinates"] = _transform_coords(geometry["coordinates"], affine_transform)
    return transformed


def _load_qgis_points_helmert(path: str) -> Affine | None:
    if not path or not os.path.exists(path):
        return None
    rows = []
    with open(path, encoding="latin1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("mapX"):
                continue
            map_x, map_y, source_x, source_y, *_ = line.split(",")
            rows.append((float(source_x), float(source_y), float(map_x), float(map_y)))
    if len(rows) < 3:
        return None
    pts = np.asarray(rows, dtype=np.float64)
    src_x = pts[:, 0]
    src_y = pts[:, 1]
    dst_x = pts[:, 2]
    dst_y = pts[:, 3]

    # 2D Helmert / similarity transform:
    # x' = a*x - b*y + tx
    # y' = b*x + a*y + ty
    n = len(pts)
    A = np.zeros((2 * n, 4), dtype=np.float64)
    A[0::2, 0] = src_x
    A[0::2, 1] = -src_y
    A[0::2, 2] = 1.0
    A[1::2, 0] = src_y
    A[1::2, 1] = src_x
    A[1::2, 3] = 1.0
    b = np.empty(2 * n, dtype=np.float64)
    b[0::2] = dst_x
    b[1::2] = dst_y

    params, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, rot, tx, ty = params
    return Affine(float(a), float(-rot), float(tx), float(rot), float(a), float(ty))


def _rasterize_polygon(geometry: dict, *, out_shape: tuple[int, int], transform) -> np.ndarray:
    return rasterize(
        [(geometry, 1)],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)


def _window_from_union_bounds(
    bounds_list: list[tuple[float, float, float, float]],
    *,
    transform,
    width: int,
    height: int,
    padding_pixels: int = 28,
) -> Window:
    left = min(b[0] for b in bounds_list)
    bottom = min(b[1] for b in bounds_list)
    right = max(b[2] for b in bounds_list)
    top = max(b[3] for b in bounds_list)
    row_top, col_left = rowcol(transform, left, top)
    row_bottom, col_right = rowcol(transform, right, bottom)
    row0 = min(row_top, row_bottom) - padding_pixels
    row1 = max(row_top, row_bottom) + padding_pixels + 1
    col0 = min(col_left, col_right) - padding_pixels
    col1 = max(col_left, col_right) + padding_pixels + 1
    row0 = max(0, row0)
    col0 = max(0, col0)
    row1 = min(height, row1)
    col1 = min(width, col1)
    return Window(col_off=col0, row_off=row0, width=col1 - col0, height=row1 - row0)


def _estimate_transform_parameter_object(
    *,
    moving_image_path: str,
    fixed_image_path: str,
    moving_band_index: int,
    fixed_band_index: int,
    use_edge_proxies: bool,
    log_to_console: bool = False,
):
    with tempfile.TemporaryDirectory(prefix="figure2_reg_") as td, rasterio.open(moving_image_path) as moving_src, rasterio.open(fixed_image_path) as fixed_src:
        fixed_data = fixed_src.read(fixed_band_index + 1).astype(np.float32)
        fixed_valid = _valid_mask(fixed_src, fixed_band_index + 1, fixed_data)
        moving_on_fixed, moving_on_fixed_valid = _reproject_band_to_fixed_grid(
            moving_src,
            fixed_src,
            moving_band_index + 1,
        )
        moving_nodata_value = _resolve_nodata(moving_src)
        fixed_nodata_value = _resolve_nodata(fixed_src)

        if use_edge_proxies:
            fixed_reg_data = _edge_proxy(fixed_data, fixed_valid)
            moving_reg_data = _edge_proxy(moving_on_fixed, moving_on_fixed_valid)
            fixed_mask = (fixed_reg_data > 0).astype(np.uint8)
            moving_mask = (moving_reg_data > 0).astype(np.uint8)
        else:
            fixed_reg_data = fixed_data
            moving_reg_data = moving_on_fixed
            fixed_mask = fixed_valid.astype(np.uint8)
            moving_mask = moving_on_fixed_valid.astype(np.uint8)

        mutual = (fixed_mask > 0) & (moving_mask > 0)
        fixed_mask = mutual.astype(np.uint8)
        moving_mask = mutual.astype(np.uint8)

        fixed_reg_path = os.path.join(td, "fixed_reg.tif")
        moving_reg_path = os.path.join(td, "moving_reg.tif")
        fixed_mask_path = os.path.join(td, "fixed_mask.tif")
        moving_mask_path = os.path.join(td, "moving_mask.tif")

        _write_single_band_tif(
            fixed_reg_path,
            fixed_reg_data.astype("float32"),
            crs=fixed_src.crs,
            transform=fixed_src.transform,
            dtype="float32",
            nodata=fixed_nodata_value,
        )
        _write_single_band_tif(
            moving_reg_path,
            moving_reg_data.astype("float32"),
            crs=fixed_src.crs,
            transform=fixed_src.transform,
            dtype="float32",
            nodata=moving_nodata_value,
        )
        _write_single_band_tif(
            fixed_mask_path,
            fixed_mask,
            crs=fixed_src.crs,
            transform=fixed_src.transform,
            dtype="uint8",
            nodata=0,
        )
        _write_single_band_tif(
            moving_mask_path,
            moving_mask,
            crs=fixed_src.crs,
            transform=fixed_src.transform,
            dtype="uint8",
            nodata=0,
        )

        return estimate_elastix_transform(
            fixed_image_path=fixed_reg_path,
            moving_image_path=moving_reg_path,
            parameter_map=DEFAULT_ALIGNMENT_PARAMETER_MAPS,
            force_nearest_resample=True,
            fixed_mask_path=fixed_mask_path,
            moving_mask_path=moving_mask_path,
            log_to_console=log_to_console,
        )


def _orthority_gcp_mask_on_fixed(
    *,
    full_moving_image_path: str,
    crop_moving_image_path: str,
    moving_polygon_geom: dict,
    gcp_geojson_path: str,
    dem_image_path: str,
    fixed_image_path: str,
    vhrharmonize_repo: str = "/home/manumea/repos/vhrharmonize",
    vhrharmonize_python: str = "/home/manumea/miniforge3/envs/vhrharmonize/bin/python",
) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="figure2_orthority_") as td:
        with rasterio.open(full_moving_image_path) as full_src, rasterio.open(crop_moving_image_path) as crop_src:
            crop_col_off = int(round((crop_src.transform.c - full_src.transform.c) / full_src.transform.a))
            crop_row_off = int(round((crop_src.transform.f - full_src.transform.f) / full_src.transform.e))
            crop_mask = _rasterize_polygon(
                moving_polygon_geom,
                out_shape=(crop_src.height, crop_src.width),
                transform=crop_src.transform,
            ).astype(np.uint8)
            mask_crop_path = os.path.join(td, "moving_crown_mask_crop.tif")
            translate_opts = gdal.TranslateOptions(
                format="GTiff",
                bandList=[1],
                outputType=gdal.GDT_Byte,
            )
            gdal.Translate(mask_crop_path, crop_moving_image_path, options=translate_opts)
            mask_ds = gdal.Open(mask_crop_path, gdal.GA_Update)
            band = mask_ds.GetRasterBand(1)
            band.WriteArray(crop_mask)
            band.SetNoDataValue(0)
            band.FlushCache()
            mask_ds.FlushCache()
            mask_ds = None

            geojson_path = os.path.join(td, "gcps.geojson")
            ortho_mask_path = os.path.join(td, "moving_crown_mask_gcp_ortho.tif")
            with open(gcp_geojson_path, "r", encoding="utf-8") as src_fp, open(geojson_path, "w", encoding="utf-8") as dst_fp:
                dst_fp.write(src_fp.read())

            ortho_script = f"""
import sys
sys.path.insert(0, {vhrharmonize_repo!r})
import orthority as oty
from osgeo import gdal
from vhrharmonize.preprocess.orthorectification import gcp_refined_rpc_orthorectification

cameras = oty.RpcCameras.from_images([{full_moving_image_path!r}])
cameras.refine({geojson_path!r})
camera = cameras.get({full_moving_image_path!r})._rpc

ds = gdal.Open({mask_crop_path!r}, gdal.GA_Update)
ds.SetMetadata({{
    "LINE_OFF": str(camera.line_off - {crop_row_off}),
    "LINE_SCALE": str(camera.line_scale),
    "SAMP_OFF": str(camera.samp_off - {crop_col_off}),
    "SAMP_SCALE": str(camera.samp_scale),
    "LAT_OFF": str(camera.lat_off),
    "LAT_SCALE": str(camera.lat_scale),
    "LONG_OFF": str(camera.long_off),
    "LONG_SCALE": str(camera.long_scale),
    "HEIGHT_OFF": str(camera.height_off),
    "HEIGHT_SCALE": str(camera.height_scale),
    "LINE_NUM_COEFF": " ".join(map(str, camera.line_num_coeff)),
    "LINE_DEN_COEFF": " ".join(map(str, camera.line_den_coeff)),
    "SAMP_NUM_COEFF": " ".join(map(str, camera.samp_num_coeff)),
    "SAMP_DEN_COEFF": " ".join(map(str, camera.samp_den_coeff)),
}}, "RPC")
ds = None

gcp_refined_rpc_orthorectification(
    input_image_path={mask_crop_path!r},
    output_image_path={ortho_mask_path!r},
    dem_image_path={dem_image_path!r},
    output_epsg=6635,
    gcp_geojson_file_path=None,
    output_nodata_value=0,
    dtype='Byte',
    output_resolution=0.312,
)
"""
            subprocess.run([vhrharmonize_python, "-c", ortho_script], check=True)

        with rasterio.open(ortho_mask_path) as gcp_src, rasterio.open(fixed_image_path) as fixed_src:
            gcp_mask_full_f = np.zeros((fixed_src.height, fixed_src.width), dtype=np.float32)
            reproject(
                source=rasterio.band(gcp_src, 1),
                destination=gcp_mask_full_f,
                src_transform=gcp_src.transform,
                src_crs=gcp_src.crs,
                dst_transform=fixed_src.transform,
                dst_crs=fixed_src.crs,
                src_nodata=0.0,
                dst_nodata=0.0,
                resampling=Resampling.nearest,
            )
            return gcp_mask_full_f > 0.5


def _panel_svg(panel: OverlayPanel, *, x: int, y: int, width: int, height: int) -> str:
    png_data = base64.b64encode(_png_bytes(panel.image)).decode("ascii")
    label_y = y - 14
    title_x = x + 28
    badge = ""
    if panel.metric_text:
        badge_x = x + width - 10
        badge_y = y + 21
        badge_rect_x = x + width - 130
        badge_rect_y = y + 8
        badge = (
            f'<rect x="{badge_rect_x}" y="{badge_rect_y}" width="120" height="24" rx="4" ry="4" '
            'fill="white" fill-opacity="0.88" stroke="#222222" stroke-width="0.8"/>\n'
            f'<text x="{badge_x}" y="{badge_y}" text-anchor="end" font-size="14" '
            'font-family="Helvetica, Arial, sans-serif" font-weight="700" fill="#111111">'
            f"{escape(panel.metric_text)}</text>\n"
        )
    return (
        f'<text x="{x}" y="{label_y}" font-size="18" font-weight="700" font-family="Helvetica, Arial, sans-serif">{escape(panel.label)}</text>\n'
        f'<text x="{title_x}" y="{label_y}" font-size="16" font-family="Helvetica, Arial, sans-serif">{escape(panel.title)}</text>\n'
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="#ffffff" stroke="#111111" stroke-width="1"/>\n'
        f'<image x="{x}" y="{y}" width="{width}" height="{height}" href="data:image/png;base64,{png_data}"/>\n'
        f'{badge}'
    )


def _legend_svg(*, x: int, y: int) -> str:
    items = [
        ("Fixed crown boundary", "#eb871e"),
        ("Moving crown boundary", "#0fbeeb"),
        ("Boundary overlap", "#ffff78"),
    ]
    body = []
    offsets = [0, 190, 378]
    for (label, color), dx in zip(items, offsets):
        xx = x + dx
        body.append(f'<rect x="{xx}" y="{y - 10}" width="14" height="14" fill="{color}" stroke="#222222" stroke-width="0.6"/>')
        body.append(
            f'<text x="{xx + 22}" y="{y + 1}" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="#333333">{escape(label)}</text>'
        )
    return "\n".join(body)


def _write_svg(path: str, *, panels: list[OverlayPanel], panel_width: int, panel_height: int, include_legend: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    gutter_x = 36
    gutter_y = 44
    margin_left = 36
    margin_top = 36
    cols = 2
    rows = int(np.ceil(len(panels) / cols))
    width = margin_left * 2 + panel_width * cols + gutter_x * (cols - 1)
    legend_band = 26 if include_legend else 0
    height = margin_top + panel_height * rows + gutter_y * (rows - 1) + 24 + legend_band
    positions = []
    for idx in range(len(panels)):
        row = idx // cols
        col = idx % cols
        positions.append(
            (
                margin_left + col * (panel_width + gutter_x),
                margin_top + row * (panel_height + gutter_y),
            )
        )
    body = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for panel, (x, y) in zip(panels, positions):
        body.append(_panel_svg(panel, x=x, y=y, width=panel_width, height=panel_height))
    if include_legend:
        legend_y = height - 12
        body.append(_legend_svg(x=margin_left, y=legend_y))
    body.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(body))


def _build_preview(panels: list[OverlayPanel], *, panel_width: int, panel_height: int, include_legend: bool = False) -> np.ndarray:
    gutter = 16
    cols = 2
    rows = int(np.ceil(len(panels) / cols))
    label_band = 22
    legend_band = 24 if include_legend else 0
    canvas = np.full(
        (3, label_band + panel_height * rows + gutter * (rows + 1) + legend_band, panel_width * cols + gutter * (cols + 1)),
        255,
        dtype=np.uint8,
    )
    positions = []
    for idx in range(len(panels)):
        row = idx // cols
        col = idx % cols
        positions.append(
            (
                label_band + gutter + row * (panel_height + gutter),
                gutter + col * (panel_width + gutter),
            )
        )
    for panel, (top, left) in zip(panels, positions):
        canvas[:, top : top + panel_height, left : left + panel_width] = panel.image
        _draw_bitmap_text(canvas, text=panel.label, x=left, y=top - 12, scale=2)
        _draw_bitmap_text(canvas, text=panel.title.upper(), x=left + 22, y=top - 12, scale=1, color=(30, 30, 30))
        if panel.metric_text:
            badge_w = 112
            badge_h = 20
            badge_x = left + panel_width - badge_w - 6
            badge_y = top + 6
            _draw_rect(canvas, x=badge_x, y=badge_y, width=badge_w, height=badge_h, color=(255, 255, 255))
            _draw_rect(canvas, x=badge_x, y=badge_y, width=badge_w, height=1, color=(34, 34, 34))
            _draw_rect(canvas, x=badge_x, y=badge_y + badge_h - 1, width=badge_w, height=1, color=(34, 34, 34))
            _draw_rect(canvas, x=badge_x, y=badge_y, width=1, height=badge_h, color=(34, 34, 34))
            _draw_rect(canvas, x=badge_x + badge_w - 1, y=badge_y, width=1, height=badge_h, color=(34, 34, 34))
            text_w = _bitmap_text_width(panel.metric_text, scale=2, spacing=1)
            _draw_bitmap_text(
                canvas,
                text=panel.metric_text,
                x=badge_x + badge_w - 6 - text_w,
                y=badge_y + 3,
                scale=2,
            )
    if include_legend:
        base_y = canvas.shape[1] - 16
        items = [
            ("FIXED CROWN BOUNDARY", (235, 135, 30)),
            ("MOVING CROWN BOUNDARY", (15, 190, 235)),
            ("BOUNDARY OVERLAP", (255, 255, 120)),
        ]
        offsets = [gutter, gutter + 170, gutter + 340]
        for (label, color), x in zip(items, offsets):
            y = base_y
            _draw_rect(canvas, x=x, y=y, width=10, height=10, color=color)
            _draw_rect(canvas, x=x, y=y, width=10, height=1, color=(34, 34, 34))
            _draw_rect(canvas, x=x, y=y + 9, width=10, height=1, color=(34, 34, 34))
            _draw_rect(canvas, x=x, y=y, width=1, height=10, color=(34, 34, 34))
            _draw_rect(canvas, x=x + 9, y=y, width=1, height=10, color=(34, 34, 34))
            _draw_bitmap_text(canvas, text=label, x=x + 16, y=y + 1, scale=1, color=(40, 40, 40))
    return canvas


def _ensure_alignment_output(
    *,
    moving_image_path: str,
    fixed_image_path: str,
    moving_band_index: int,
    fixed_band_index: int,
    solve_resolution: float | None,
    cache_dir: str,
    force: bool,
) -> tuple[str, str]:
    os.makedirs(cache_dir, exist_ok=True)
    raw_output = os.path.join(cache_dir, "raw_registration_fixed_grid.tif")
    grad_output = os.path.join(cache_dir, "gradient_registration_fixed_grid.tif")
    if force or not os.path.exists(raw_output):
        align_image_pair(
            moving_image_path=moving_image_path,
            fixed_image_path=fixed_image_path,
            output_image_path=raw_output,
            moving_band_index=moving_band_index,
            fixed_band_index=fixed_band_index,
            output_on_moving_grid=False,
            use_edge_proxies=False,
            solve_resolution=solve_resolution,
        )
    if force or not os.path.exists(grad_output):
        align_image_pair(
            moving_image_path=moving_image_path,
            fixed_image_path=fixed_image_path,
            output_image_path=grad_output,
            moving_band_index=moving_band_index,
            fixed_band_index=fixed_band_index,
            output_on_moving_grid=False,
            use_edge_proxies=True,
            solve_resolution=solve_resolution,
        )
    return raw_output, grad_output


def _prepare_panels(
    *,
    moving_image_path: str,
    fixed_image_path: str,
    gcp_image_path: str | None,
    gcp_geojson_path: str | None,
    gcp_points_path: str | None,
    full_moving_image_path: str | None,
    orthority_dem_path: str | None,
    moving_crowns_path: str,
    fixed_crowns_path: str,
    moving_band_index: int,
    fixed_band_index: int,
    panel_width: int,
    force: bool,
) -> list[OverlayPanel]:
    del force
    moving_geom, moving_bounds = _load_single_polygon(moving_crowns_path)
    fixed_geom, fixed_bounds = _load_single_polygon(fixed_crowns_path)

    raw_transform = _estimate_transform_parameter_object(
        moving_image_path=moving_image_path,
        fixed_image_path=fixed_image_path,
        moving_band_index=moving_band_index,
        fixed_band_index=fixed_band_index,
        use_edge_proxies=False,
    )
    edge_transform = _estimate_transform_parameter_object(
        moving_image_path=moving_image_path,
        fixed_image_path=fixed_image_path,
        moving_band_index=moving_band_index,
        fixed_band_index=fixed_band_index,
        use_edge_proxies=True,
    )

    with rasterio.open(moving_image_path) as moving_src, rasterio.open(fixed_image_path) as fixed_src:
        fixed_data = fixed_src.read(fixed_band_index + 1).astype(np.float32)
        fixed_valid = _valid_mask(fixed_src, fixed_band_index + 1, fixed_data)
        unaligned_data, unaligned_valid = _reproject_band_to_fixed_grid(
            moving_src,
            fixed_src,
            moving_band_index + 1,
        )
        fixed_mask_full = _rasterize_polygon(
            fixed_geom,
            out_shape=(fixed_src.height, fixed_src.width),
            transform=fixed_src.transform,
        )
        moving_mask_src = _rasterize_polygon(
            moving_geom,
            out_shape=(moving_src.height, moving_src.width),
            transform=moving_src.transform,
        ).astype(np.float32)
        moving_mask_on_fixed = np.zeros((fixed_src.height, fixed_src.width), dtype=np.float32)
        reproject(
            source=moving_mask_src,
            destination=moving_mask_on_fixed,
            src_transform=moving_src.transform,
            src_crs=moving_src.crs,
            dst_transform=fixed_src.transform,
            dst_crs=fixed_src.crs,
            src_nodata=0.0,
            dst_nodata=0.0,
            resampling=Resampling.nearest,
        )
        raw_mask_full = apply_elastix_transform_array(moving_mask_on_fixed, raw_transform) > 0.5
        edge_mask_full = apply_elastix_transform_array(moving_mask_on_fixed, edge_transform) > 0.5
        unaligned_mask_full = moving_mask_on_fixed > 0.5

        gcp_mask_full = None
        gcp_boundary_px = None
        gcp_bounds = None
        if gcp_image_path and os.path.exists(gcp_image_path):
            with rasterio.open(gcp_image_path) as gcp_src:
                gcp_transform = _estimate_transform_parameter_object(
                    moving_image_path=moving_image_path,
                    fixed_image_path=gcp_image_path,
                    moving_band_index=moving_band_index,
                    fixed_band_index=moving_band_index,
                    use_edge_proxies=False,
                )
                moving_mask_on_gcp = np.zeros((gcp_src.height, gcp_src.width), dtype=np.float32)
                reproject(
                    source=moving_mask_src,
                    destination=moving_mask_on_gcp,
                    src_transform=moving_src.transform,
                    src_crs=moving_src.crs,
                    dst_transform=gcp_src.transform,
                    dst_crs=gcp_src.crs,
                    src_nodata=0.0,
                    dst_nodata=0.0,
                    resampling=Resampling.nearest,
                )
                gcp_mask_native = apply_elastix_transform_array(moving_mask_on_gcp, gcp_transform) > 0.5
                gcp_mask_full_f = np.zeros((fixed_src.height, fixed_src.width), dtype=np.float32)
                reproject(
                    source=gcp_mask_native.astype(np.float32),
                    destination=gcp_mask_full_f,
                    src_transform=gcp_src.transform,
                    src_crs=gcp_src.crs,
                    dst_transform=fixed_src.transform,
                    dst_crs=fixed_src.crs,
                    src_nodata=0.0,
                    dst_nodata=0.0,
                    resampling=Resampling.nearest,
                )
                gcp_mask_full = gcp_mask_full_f > 0.5
                gcp_bounds = fixed_bounds
        elif (
            full_moving_image_path
            and orthority_dem_path
            and gcp_geojson_path
            and os.path.exists(full_moving_image_path)
            and os.path.exists(orthority_dem_path)
            and os.path.exists(gcp_geojson_path)
        ):
            gcp_mask_full = _orthority_gcp_mask_on_fixed(
                full_moving_image_path=full_moving_image_path,
                crop_moving_image_path=moving_image_path,
                moving_polygon_geom=moving_geom,
                gcp_geojson_path=gcp_geojson_path,
                dem_image_path=orthority_dem_path,
                fixed_image_path=fixed_image_path,
            )
            gcp_bounds = fixed_bounds

        union_bounds = [moving_bounds, fixed_bounds]
        if gcp_bounds is not None:
            union_bounds.append(gcp_bounds)

        crop_window = _window_from_union_bounds(
            union_bounds,
            transform=fixed_src.transform,
            width=fixed_src.width,
            height=fixed_src.height,
            padding_pixels=32,
        )

        fixed_data = _crop(fixed_data, crop_window)
        fixed_valid = _crop(fixed_valid, crop_window)
        fixed_mask = _crop(fixed_mask_full, crop_window)
        unaligned_mask = _crop(unaligned_mask_full, crop_window)
        raw_mask = _crop(raw_mask_full, crop_window)
        edge_mask = _crop(edge_mask_full, crop_window)
        if gcp_mask_full is not None:
            gcp_mask = _crop(gcp_mask_full, crop_window)

        unaligned_panel_base = _overlay_crown_panel(fixed_data, fixed_valid, fixed_mask, unaligned_mask)
        raw_panel_base = _overlay_crown_panel(fixed_data, fixed_valid, fixed_mask, raw_mask)
        edge_panel_base = _overlay_crown_panel(fixed_data, fixed_valid, fixed_mask, edge_mask)
        if gcp_mask_full is not None:
            gcp_panel_base = _overlay_crown_panel(fixed_data, fixed_valid, fixed_mask, gcp_mask)

        unaligned_boundary_px = _symmetric_boundary_rmse(fixed_mask, unaligned_mask)
        raw_boundary_px = _symmetric_boundary_rmse(fixed_mask, raw_mask)
        edge_boundary_px = _symmetric_boundary_rmse(fixed_mask, edge_mask)
        if gcp_mask_full is not None:
            gcp_boundary_px = _symmetric_boundary_rmse(fixed_mask, gcp_mask)
        fixed_pixel_size_m = float(abs(fixed_src.transform.a))
        unaligned_boundary_m = unaligned_boundary_px * fixed_pixel_size_m
        raw_boundary_m = raw_boundary_px * fixed_pixel_size_m
        edge_boundary_m = edge_boundary_px * fixed_pixel_size_m
        if gcp_mask_full is not None and gcp_boundary_px is not None:
            gcp_boundary_m = gcp_boundary_px * fixed_pixel_size_m

        full_height = fixed_data.shape[0]
        full_width = fixed_data.shape[1]
        panel_height = max(1, int(round(panel_width * full_height / full_width)))
        unaligned_panel = _resize_rgb(unaligned_panel_base, panel_height, panel_width)
        raw_panel = _resize_rgb(raw_panel_base, panel_height, panel_width)
        edge_panel = _resize_rgb(edge_panel_base, panel_height, panel_width)
        if gcp_mask_full is not None:
            gcp_panel = _resize_rgb(gcp_panel_base, panel_height, panel_width)

        panels = [OverlayPanel("A", "Default RPC", unaligned_panel, metric_text=f"RMSE: {unaligned_boundary_m:.2f} m")]
        if gcp_mask_full is not None and gcp_boundary_px is not None:
            panels.append(OverlayPanel("B", "Manual GCP", gcp_panel, metric_text=f"RMSE: {gcp_boundary_m:.2f} m"))
        panels.extend(
            [
                OverlayPanel("C", "Raw-intensity registration", raw_panel, metric_text=f"RMSE: {raw_boundary_m:.2f} m"),
                OverlayPanel("D", "Gradient-Magnitude registration", edge_panel, metric_text=f"RMSE: {edge_boundary_m:.2f} m"),
            ]
        )
        return panels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moving-image", default=DEFAULT_MOVING_IMAGE)
    parser.add_argument("--fixed-image", default=DEFAULT_FIXED_IMAGE)
    parser.add_argument("--gcp-image", default=DEFAULT_GCP_IMAGE)
    parser.add_argument("--gcp-geojson", default=DEFAULT_GCP_GEOJSON)
    parser.add_argument("--gcp-points", default=DEFAULT_GCP_POINTS)
    parser.add_argument("--full-moving-image", default=DEFAULT_FULL_MOVING_IMAGE)
    parser.add_argument("--orthority-dem", default=DEFAULT_ORTHORITY_DEM)
    parser.add_argument("--moving-crowns", default=DEFAULT_MOVING_CROWNS)
    parser.add_argument("--fixed-crowns", default=DEFAULT_FIXED_CROWNS)
    parser.add_argument("--moving-band-index", type=int, default=5)
    parser.add_argument("--fixed-band-index", type=int, default=0)
    parser.add_argument("--panel-width", type=int, default=320)
    parser.add_argument("--output-svg", default=DEFAULT_OUTPUT_SVG)
    parser.add_argument("--output-preview", default=DEFAULT_OUTPUT_PREVIEW)
    parser.add_argument("--include-legend", action="store_true")
    parser.add_argument("--force", action="store_true", help="Retained for CLI compatibility; currently ignored.")
    args = parser.parse_args()

    panels = _prepare_panels(
        moving_image_path=args.moving_image,
        fixed_image_path=args.fixed_image,
        gcp_image_path=args.gcp_image,
        gcp_geojson_path=args.gcp_geojson,
        gcp_points_path=args.gcp_points,
        full_moving_image_path=args.full_moving_image,
        orthority_dem_path=args.orthority_dem,
        moving_crowns_path=args.moving_crowns,
        fixed_crowns_path=args.fixed_crowns,
        moving_band_index=args.moving_band_index,
        fixed_band_index=args.fixed_band_index,
        panel_width=args.panel_width,
        force=args.force,
    )
    panel_height = panels[0].image.shape[1]
    _write_svg(
        args.output_svg,
        panels=panels,
        panel_width=args.panel_width,
        panel_height=panel_height,
        include_legend=args.include_legend,
    )
    _write_png(
        args.output_preview,
        _build_preview(
            panels,
            panel_width=args.panel_width,
            panel_height=panel_height,
            include_legend=args.include_legend,
        ),
    )
    print(args.output_svg)
    print(args.output_preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
