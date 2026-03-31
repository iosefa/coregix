#!/usr/bin/env python3
"""Create Figure 1 comparing raw and edge-proxy optical/LiDAR imagery."""

from __future__ import annotations

import argparse
import base64
import os
import warnings
from dataclasses import dataclass
from html import escape
from typing import Iterable

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from rasterio.warp import reproject
from rasterio.windows import Window

from coregix.pipelines.alignment import _edge_proxy


DEFAULT_MOVING_IMAGE = "examples/data/moving/img_sat_lg.tif"
DEFAULT_FIXED_IMAGE = "examples/data/fixed/intensity_lg.tif"
DEFAULT_OUTPUT_SVG = "manuscript/figures/figure1_raw_vs_gradient.svg"
DEFAULT_OUTPUT_PREVIEW = "manuscript/figures/figure1_raw_vs_gradient_preview.png"


@dataclass
class Panel:
    label: str
    title: str
    image: np.ndarray


def _resolve_nodata(src: rasterio.DatasetReader) -> float | None:
    return src.nodata


def _valid_mask(src: rasterio.DatasetReader, band_index_1based: int, data: np.ndarray) -> np.ndarray:
    mask = src.read_masks(band_index_1based) > 0
    nodata = _resolve_nodata(src)
    if nodata is not None:
        mask &= data != nodata
    return mask


def _reproject_to_fixed_grid(
    moving_src: rasterio.DatasetReader,
    fixed_src: rasterio.DatasetReader,
    band_index_1based: int,
) -> tuple[np.ndarray, np.ndarray]:
    moving_data = moving_src.read(band_index_1based).astype(np.float32)
    moving_valid = _valid_mask(moving_src, band_index_1based, moving_data)
    moving_nodata = _resolve_nodata(moving_src)
    fill_value = moving_nodata if moving_nodata is not None else 0.0

    moving_on_fixed = np.full((fixed_src.height, fixed_src.width), fill_value, dtype=np.float32)
    moving_valid_reprojected = np.zeros((fixed_src.height, fixed_src.width), dtype=np.uint8)
    reproject(
        source=moving_data,
        destination=moving_on_fixed,
        src_transform=moving_src.transform,
        src_crs=moving_src.crs,
        dst_transform=fixed_src.transform,
        dst_crs=fixed_src.crs,
        src_nodata=moving_nodata,
        dst_nodata=fill_value,
        resampling=Resampling.nearest,
    )
    reproject(
        source=moving_valid.astype(np.uint8),
        destination=moving_valid_reprojected,
        src_transform=moving_src.transform,
        src_crs=moving_src.crs,
        dst_transform=fixed_src.transform,
        dst_crs=fixed_src.crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    return moving_on_fixed, moving_valid_reprojected > 0


def _percentile_scale(data: np.ndarray, valid_mask: np.ndarray, *, low: float, high: float) -> np.ndarray:
    out = np.full(data.shape, 255, dtype=np.uint8)
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


def _choose_crop(
    fixed_edge: np.ndarray,
    moving_edge: np.ndarray,
    mutual_valid: np.ndarray,
    *,
    crop_size: int,
) -> Window:
    height, width = mutual_valid.shape
    crop_h = min(crop_size, height)
    crop_w = min(crop_size, width)
    if crop_h == height and crop_w == width:
        return Window(0, 0, width, height)

    energy = np.where(mutual_valid, fixed_edge + moving_edge, 0.0).astype(np.float64)
    valid_counts = mutual_valid.astype(np.float64)
    energy_integral = _integral_image(energy)
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
            score = _window_sum(energy_integral, row_off, col_off, crop_h, crop_w)
            if score > best_score:
                best_score = score
                best = Window(col_off, row_off, crop_w, crop_h)

    return best


def _crop(arr: np.ndarray, window: Window) -> np.ndarray:
    row0 = int(window.row_off)
    row1 = row0 + int(window.height)
    col0 = int(window.col_off)
    col1 = col0 + int(window.width)
    return arr[row0:row1, col0:col1]


def _rgb_from_gray(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray, gray, gray], axis=0)


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


def _build_preview(panels: Iterable[Panel], panel_px: int = 320, gutter: int = 16) -> np.ndarray:
    panels = list(panels)
    if len(panels) != 4:
        raise ValueError("Expected exactly four panels for the preview mosaic.")
    canvas_h = panel_px * 2 + gutter * 3
    canvas_w = panel_px * 2 + gutter * 3
    canvas = np.full((3, canvas_h, canvas_w), 255, dtype=np.uint8)
    positions = [
        (gutter, gutter),
        (gutter, panel_px + 2 * gutter),
        (panel_px + 2 * gutter, gutter),
        (panel_px + 2 * gutter, panel_px + 2 * gutter),
    ]
    for panel, (top, left) in zip(panels, positions):
        rgb = panel.image
        canvas[:, top : top + panel_px, left : left + panel_px] = rgb
    return canvas


def _panel_svg(panel: Panel, *, x: int, y: int, size: int) -> str:
    png_data = base64.b64encode(_png_bytes(panel.image)).decode("ascii")
    title_y = y - 14
    label_y = y - 14
    title_x = x + 28
    return (
        f'<text x="{x}" y="{label_y}" font-size="18" font-weight="700" font-family="Helvetica, Arial, sans-serif">'
        f"{escape(panel.label)}</text>\n"
        f'<text x="{title_x}" y="{title_y}" font-size="16" font-family="Helvetica, Arial, sans-serif">'
        f"{escape(panel.title)}</text>\n"
        f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="#ffffff" stroke="#111111" stroke-width="1"/>\n'
        f'<image x="{x}" y="{y}" width="{size}" height="{size}" href="data:image/png;base64,{png_data}"/>\n'
    )


def _write_svg(
    path: str,
    *,
    panels: list[Panel],
    moving_band_index: int,
    panel_px: int = 320,
    gutter_x: int = 36,
    gutter_y: int = 44,
    margin_left: int = 36,
    margin_top: int = 44,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width = margin_left * 2 + panel_px * 2 + gutter_x
    height = margin_top + panel_px * 2 + gutter_y + 64
    positions = [
        (margin_left, margin_top),
        (margin_left + panel_px + gutter_x, margin_top),
        (margin_left, margin_top + panel_px + gutter_y),
        (margin_left + panel_px + gutter_x, margin_top + panel_px + gutter_y),
    ]

    body = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            '<text x="36" y="24" font-size="18" font-family="Helvetica, Arial, sans-serif" font-weight="700">'
            'Figure 1. Raw optical and LiDAR intensity versus gradient magnitude'
            "</text>"
        ),
    ]
    for panel, (x, y) in zip(panels, positions):
        body.append(_panel_svg(panel, x=x, y=y, size=panel_px))
    body.append(
        (
            f'<text x="36" y="{height - 16}" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="#333333">'
            f"Optical band {moving_band_index + 1} is resampled to the LiDAR grid. Edge-proxy images use coregix._edge_proxy exactly."
            "</text>"
        )
    )
    body.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(body))


def _prepare_panels(
    moving_image_path: str,
    fixed_image_path: str,
    *,
    moving_band_index: int,
    crop_size: int,
) -> list[Panel]:
    with rasterio.open(moving_image_path) as moving_src, rasterio.open(fixed_image_path) as fixed_src:
        moving_band = moving_band_index + 1
        fixed_data = fixed_src.read(1).astype(np.float32)
        fixed_valid = _valid_mask(fixed_src, 1, fixed_data)
        moving_on_fixed, moving_valid = _reproject_to_fixed_grid(moving_src, fixed_src, moving_band)
        mutual_valid = fixed_valid & moving_valid

        fixed_edge = _edge_proxy(fixed_data, fixed_valid)
        moving_edge = _edge_proxy(moving_on_fixed, moving_valid)
        crop_window = _choose_crop(fixed_edge, moving_edge, mutual_valid, crop_size=crop_size)

        moving_raw_crop = _crop(moving_on_fixed, crop_window)
        fixed_raw_crop = _crop(fixed_data, crop_window)
        moving_edge_crop = _crop(moving_edge, crop_window)
        fixed_edge_crop = _crop(fixed_edge, crop_window)
        mutual_crop = _crop(mutual_valid, crop_window)

        moving_raw_img = _rgb_from_gray(_percentile_scale(moving_raw_crop, mutual_crop, low=2, high=98))
        fixed_raw_img = _rgb_from_gray(_percentile_scale(fixed_raw_crop, mutual_crop, low=2, high=98))
        moving_edge_img = _rgb_from_gray(_percentile_scale(moving_edge_crop, mutual_crop, low=2, high=99))
        fixed_edge_img = _rgb_from_gray(_percentile_scale(fixed_edge_crop, mutual_crop, low=2, high=99))

        return [
            Panel("A", f"Optical band {moving_band} (raw, resampled)", moving_raw_img),
            Panel("B", "LiDAR intensity (raw)", fixed_raw_img),
            Panel("C", f"Optical band {moving_band} gradient magnitude", moving_edge_img),
            Panel("D", "LiDAR gradient magnitude", fixed_edge_img),
        ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moving-image", default=DEFAULT_MOVING_IMAGE)
    parser.add_argument("--fixed-image", default=DEFAULT_FIXED_IMAGE)
    parser.add_argument("--moving-band-index", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--output-svg", default=DEFAULT_OUTPUT_SVG)
    parser.add_argument("--output-preview", default=DEFAULT_OUTPUT_PREVIEW)
    args = parser.parse_args()

    panels = _prepare_panels(
        args.moving_image,
        args.fixed_image,
        moving_band_index=args.moving_band_index,
        crop_size=args.crop_size,
    )
    _write_svg(
        args.output_svg,
        panels=panels,
        moving_band_index=args.moving_band_index,
        panel_px=int(args.crop_size),
    )
    _write_png(args.output_preview, _build_preview(panels, panel_px=int(args.crop_size)))
    print(args.output_svg)
    print(args.output_preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
