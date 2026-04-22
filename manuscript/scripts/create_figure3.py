#!/usr/bin/env python3
"""Create Figure 3: RMSE boxplot across all crowns."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
from osgeo import ogr
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from coregix.preprocess.registration import apply_elastix_transform_array
from create_figure2 import (
    DEFAULT_FIXED_CROWNS,
    DEFAULT_FIXED_IMAGE,
    DEFAULT_GCP_IMAGE,
    DEFAULT_MOVING_CROWNS,
    DEFAULT_MOVING_IMAGE,
    _bitmap_text_width,
    _draw_bitmap_text,
    _draw_rect,
    _estimate_transform_parameter_object,
    _rasterize_polygon,
    _symmetric_boundary_rmse,
    _write_png,
)

DEFAULT_DENSE_MOVING_IMAGE = "manuscript/data/moving_dense.tif"
DEFAULT_DENSE_FIXED_IMAGE = "manuscript/data/fixed_dense.tif"
DEFAULT_DENSE_GCP_IMAGE = "manuscript/data/gcp_dense.tif"
DEFAULT_DENSE_MOVING_CROWNS = "manuscript/data/crowns_moving_dense.gpkg"
DEFAULT_DENSE_FIXED_CROWNS = "manuscript/data/crowns_fixed_dense.gpkg"
DEFAULT_OUTPUT_SVG = "manuscript/figures/figure3_rmse_boxplot.svg"
DEFAULT_OUTPUT_PREVIEW = "manuscript/figures/figure3_rmse_boxplot_preview.png"


@dataclass
class MethodSeries:
    label: str
    group: str
    color: tuple[int, int, int]
    values: np.ndarray
    crown_ids: list[int]


def _load_polygons_by_id(path: str, *, id_field: str = "crown_id") -> dict[int, dict]:
    ogr.DontUseExceptions()
    ds = ogr.Open(path)
    if ds is None:
        raise RuntimeError(f"Failed to open vector file: {path}")
    layer = ds.GetLayer(0)
    if layer is None:
        raise RuntimeError(f"No layer found in {path}")
    out: dict[int, dict] = {}
    for feat in layer:
        crown_id = feat.GetField(id_field)
        geom = feat.GetGeometryRef()
        if crown_id is None or geom is None:
            continue
        out[int(crown_id)] = json.loads(geom.ExportToJson())
    ds = None
    if not out:
        raise RuntimeError(f"No crown geometries found in {path}")
    return out


def _box_stats(values: np.ndarray) -> dict[str, float]:
    vals = np.sort(np.asarray(values, dtype=np.float64))
    q1 = float(np.percentile(vals, 25))
    median = float(np.percentile(vals, 50))
    q3 = float(np.percentile(vals, 75))
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    whisker_low = float(vals[vals >= lower_fence][0])
    whisker_high = float(vals[vals <= upper_fence][-1])
    return {
        "min": float(vals.min()),
        "max": float(vals.max()),
        "q1": q1,
        "median": median,
        "q3": q3,
        "whisker_low": whisker_low,
        "whisker_high": whisker_high,
    }


def _compute_method_rmse_series(
    *,
    moving_image_path: str,
    fixed_image_path: str,
    gcp_image_path: str,
    moving_crowns_path: str,
    fixed_crowns_path: str,
    moving_band_index: int,
) -> list[MethodSeries]:
    moving_polys = _load_polygons_by_id(moving_crowns_path)
    fixed_polys = _load_polygons_by_id(fixed_crowns_path)
    common_ids = sorted(set(moving_polys) & set(fixed_polys))
    if not common_ids:
        raise RuntimeError("No shared crown_id values between moving and fixed crown files")

    edge_transform = _estimate_transform_parameter_object(
        moving_image_path=moving_image_path,
        fixed_image_path=fixed_image_path,
        moving_band_index=moving_band_index,
        fixed_band_index=0,
        use_edge_proxies=True,
    )
    gcp_transform = _estimate_transform_parameter_object(
        moving_image_path=moving_image_path,
        fixed_image_path=gcp_image_path,
        moving_band_index=moving_band_index,
        fixed_band_index=moving_band_index,
        use_edge_proxies=False,
    )

    no_alignment_vals: list[float] = []
    gcp_vals: list[float] = []
    edge_vals: list[float] = []

    with rasterio.open(moving_image_path) as moving_src, rasterio.open(fixed_image_path) as fixed_src, rasterio.open(
        gcp_image_path
    ) as gcp_src:
        fixed_pixel_size_m = float(abs(fixed_src.transform.a))
        for crown_id in common_ids:
            moving_geom = moving_polys[crown_id]
            fixed_geom = fixed_polys[crown_id]

            fixed_mask = _rasterize_polygon(
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

            unaligned_mask = moving_mask_on_fixed > 0.5
            edge_mask = apply_elastix_transform_array(moving_mask_on_fixed, edge_transform) > 0.5
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
            gcp_mask = gcp_mask_full_f > 0.5

            no_alignment_vals.append(_symmetric_boundary_rmse(fixed_mask, unaligned_mask) * fixed_pixel_size_m)
            gcp_vals.append(_symmetric_boundary_rmse(fixed_mask, gcp_mask) * fixed_pixel_size_m)
            edge_vals.append(_symmetric_boundary_rmse(fixed_mask, edge_mask) * fixed_pixel_size_m)

    return [
        MethodSeries(
            "Dry Forest",
            "Default RPC",
            (110, 110, 110),
            np.asarray(no_alignment_vals, dtype=np.float64),
            list(common_ids),
        ),
        MethodSeries(
            "Dry Forest",
            "Manual GCP",
            (214, 126, 44),
            np.asarray(gcp_vals, dtype=np.float64),
            list(common_ids),
        ),
        MethodSeries(
            "Dry Forest",
            "Gradient-Magnitude",
            (26, 140, 255),
            np.asarray(edge_vals, dtype=np.float64),
            list(common_ids),
        ),
    ]


def _value_to_y(value: float, *, y_min: float, y_max: float, plot_top: int, plot_height: int) -> float:
    if y_max <= y_min:
        return float(plot_top + plot_height)
    frac = (value - y_min) / (y_max - y_min)
    return float(plot_top + plot_height - frac * plot_height)


def _svg_boxplot(path: str, series_list: list[MethodSeries]) -> None:
    width = 840
    height = 520
    margin_left = 68
    margin_right = 16
    margin_top = 36
    margin_bottom = 108
    plot_left = margin_left
    plot_top = margin_top
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    values_all = np.concatenate([s.values for s in series_list])
    y_max = float(np.ceil((values_all.max() * 1.1) / 0.1) * 0.1)
    y_min = 0.0
    tick_step = 0.5 if y_max > 2.0 else 0.25
    groups = []
    for series in series_list:
        if series.group not in groups:
            groups.append(series.group)
    group_centers = np.linspace(plot_left + plot_width * 0.22, plot_left + plot_width * 0.78, len(groups))
    pair_offset = plot_width / 14
    box_width = plot_width / 19
    x_lookup: dict[tuple[str, str], float] = {}
    for center, group in zip(group_centers, groups):
        x_lookup[(group, "Wet Forest")] = center - pair_offset / 2
        x_lookup[(group, "Dry Forest")] = center + pair_offset / 2

    body = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_height}" stroke="#222222" stroke-width="1.2"/>',
        f'<line x1="{plot_left}" y1="{plot_top + plot_height}" x2="{plot_left + plot_width}" y2="{plot_top + plot_height}" stroke="#222222" stroke-width="1.2"/>',
        f'<text x="28" y="{plot_top + plot_height / 2}" font-size="16" font-family="Helvetica, Arial, sans-serif" transform="rotate(-90 28 {plot_top + plot_height / 2})">Boundary RMSE (m)</text>',
    ]

    tick = y_min
    while tick <= y_max + 1e-9:
        y = _value_to_y(tick, y_min=y_min, y_max=y_max, plot_top=plot_top, plot_height=plot_height)
        body.append(f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_left + plot_width}" y2="{y:.2f}" stroke="#e2e2e2" stroke-width="1"/>')
        body.append(f'<text x="{plot_left - 10}" y="{y + 5:.2f}" text-anchor="end" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="#444444">{tick:.2f}</text>')
        tick += tick_step

    for series in series_list:
        stats = _box_stats(series.values)
        x = x_lookup[(series.group, series.label)]
        color = f"rgb({series.color[0]},{series.color[1]},{series.color[2]})"
        q1_y = _value_to_y(stats["q1"], y_min=y_min, y_max=y_max, plot_top=plot_top, plot_height=plot_height)
        q3_y = _value_to_y(stats["q3"], y_min=y_min, y_max=y_max, plot_top=plot_top, plot_height=plot_height)
        med_y = _value_to_y(stats["median"], y_min=y_min, y_max=y_max, plot_top=plot_top, plot_height=plot_height)
        low_y = _value_to_y(stats["whisker_low"], y_min=y_min, y_max=y_max, plot_top=plot_top, plot_height=plot_height)
        high_y = _value_to_y(stats["whisker_high"], y_min=y_min, y_max=y_max, plot_top=plot_top, plot_height=plot_height)

        for j, value in enumerate(series.values):
            jitter = ((j % 7) - 3) * (box_width / 14.0)
            py = _value_to_y(float(value), y_min=y_min, y_max=y_max, plot_top=plot_top, plot_height=plot_height)
            body.append(
                f'<circle cx="{x + jitter:.2f}" cy="{py:.2f}" r="3.0" fill="{color}" fill-opacity="0.35" stroke="none"/>'
            )

        body.append(
            f'<line x1="{x:.2f}" y1="{high_y:.2f}" x2="{x:.2f}" y2="{q3_y:.2f}" stroke="{color}" stroke-width="2"/>'
        )
        body.append(
            f'<line x1="{x:.2f}" y1="{q1_y:.2f}" x2="{x:.2f}" y2="{low_y:.2f}" stroke="{color}" stroke-width="2"/>'
        )
        body.append(
            f'<line x1="{x - box_width / 2:.2f}" y1="{high_y:.2f}" x2="{x + box_width / 2:.2f}" y2="{high_y:.2f}" stroke="{color}" stroke-width="2"/>'
        )
        body.append(
            f'<line x1="{x - box_width / 2:.2f}" y1="{low_y:.2f}" x2="{x + box_width / 2:.2f}" y2="{low_y:.2f}" stroke="{color}" stroke-width="2"/>'
        )
        body.append(
            f'<rect x="{x - box_width / 2:.2f}" y="{q3_y:.2f}" width="{box_width:.2f}" height="{max(1.0, q1_y - q3_y):.2f}" fill="{color}" fill-opacity="0.28" stroke="{color}" stroke-width="2"/>'
        )
        body.append(
            f'<line x1="{x - box_width / 2:.2f}" y1="{med_y:.2f}" x2="{x + box_width / 2:.2f}" y2="{med_y:.2f}" stroke="{color}" stroke-width="3"/>'
        )
    for center, group in zip(group_centers, groups):
        body.append(
            f'<text x="{center:.2f}" y="{plot_top + plot_height + 28}" text-anchor="middle" font-size="13" font-family="Helvetica, Arial, sans-serif">{group}</text>'
        )

    legend_x = width - 245
    legend_y = 42
    legend_items = [
        ("Dry Forest", "rgb(194,118,44)"),
        ("Wet Forest", "rgb(38,132,78)"),
    ]
    for idx, (label, color) in enumerate(legend_items):
        y = legend_y + idx * 24
        body.append(f'<rect x="{legend_x}" y="{y - 10}" width="14" height="14" fill="{color}" stroke="none"/>')
        body.append(
            f'<text x="{legend_x + 22}" y="{y + 1}" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="#333333">{label}</text>'
        )

    body.append("</svg>")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(body))


def _preview_boxplot(path: str, series_list: list[MethodSeries]) -> None:
    width = 840
    height = 520
    margin_left = 68
    margin_right = 16
    margin_top = 36
    margin_bottom = 108
    plot_left = margin_left
    plot_top = margin_top
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    values_all = np.concatenate([s.values for s in series_list])
    y_max = float(np.ceil((values_all.max() * 1.1) / 0.1) * 0.1)
    y_min = 0.0
    tick_step = 0.5 if y_max > 2.0 else 0.25
    groups = []
    for series in series_list:
        if series.group not in groups:
            groups.append(series.group)
    group_centers = np.linspace(plot_left + plot_width * 0.22, plot_left + plot_width * 0.78, len(groups))
    pair_offset = plot_width / 14
    box_width = plot_width / 19
    x_lookup: dict[tuple[str, str], int] = {}
    for center, group in zip(group_centers, groups):
        x_lookup[(group, "Wet Forest")] = int(round(center - pair_offset / 2))
        x_lookup[(group, "Dry Forest")] = int(round(center + pair_offset / 2))

    canvas = np.full((3, height, width), 255, dtype=np.uint8)
    _draw_rect(canvas, x=plot_left, y=plot_top, width=2, height=plot_height, color=(34, 34, 34))
    _draw_rect(canvas, x=plot_left, y=plot_top + plot_height, width=plot_width, height=2, color=(34, 34, 34))

    tick = y_min
    while tick <= y_max + 1e-9:
        y = int(round(_value_to_y(tick, y_min=y_min, y_max=y_max, plot_top=plot_top, plot_height=plot_height)))
        _draw_rect(canvas, x=plot_left, y=y, width=plot_width, height=1, color=(226, 226, 226))
        tick_text = f"{tick:.2f}"
        _draw_bitmap_text(canvas, text=tick_text, x=plot_left - 8 - _bitmap_text_width(tick_text, scale=2), y=y - 6, scale=2, color=(70, 70, 70))
        tick += tick_step

    y_label = "RMSE M"
    _draw_bitmap_text(canvas, text=y_label, x=20, y=16, scale=2, color=(40, 40, 40))

    for series in series_list:
        stats = _box_stats(series.values)
        x = x_lookup[(series.group, series.label)]
        color = series.color
        q1_y = int(round(_value_to_y(stats["q1"], y_min=y_min, y_max=y_max, plot_top=plot_top, plot_height=plot_height)))
        q3_y = int(round(_value_to_y(stats["q3"], y_min=y_min, y_max=y_max, plot_top=plot_top, plot_height=plot_height)))
        med_y = int(round(_value_to_y(stats["median"], y_min=y_min, y_max=y_max, plot_top=plot_top, plot_height=plot_height)))
        low_y = int(round(_value_to_y(stats["whisker_low"], y_min=y_min, y_max=y_max, plot_top=plot_top, plot_height=plot_height)))
        high_y = int(round(_value_to_y(stats["whisker_high"], y_min=y_min, y_max=y_max, plot_top=plot_top, plot_height=plot_height)))
        half_w = int(round(box_width / 2))

        for j, value in enumerate(series.values):
            jitter = ((j % 7) - 3) * max(1, int(round(box_width / 14.0)))
            py = int(round(_value_to_y(float(value), y_min=y_min, y_max=y_max, plot_top=plot_top, plot_height=plot_height)))
            _draw_rect(canvas, x=x + jitter - 2, y=py - 2, width=4, height=4, color=color)

        _draw_rect(canvas, x=x - 1, y=high_y, width=2, height=max(1, q3_y - high_y), color=color)
        _draw_rect(canvas, x=x - 1, y=q1_y, width=2, height=max(1, low_y - q1_y), color=color)
        _draw_rect(canvas, x=x - half_w, y=high_y - 1, width=2 * half_w, height=2, color=color)
        _draw_rect(canvas, x=x - half_w, y=low_y - 1, width=2 * half_w, height=2, color=color)
        _draw_rect(canvas, x=x - half_w, y=q3_y, width=2 * half_w, height=max(1, q1_y - q3_y), color=(235, 235, 235))
        _draw_rect(canvas, x=x - half_w, y=q3_y, width=2 * half_w, height=2, color=color)
        _draw_rect(canvas, x=x - half_w, y=q1_y - 1, width=2 * half_w, height=2, color=color)
        _draw_rect(canvas, x=x - half_w, y=q3_y, width=2, height=max(1, q1_y - q3_y), color=color)
        _draw_rect(canvas, x=x + half_w - 1, y=q3_y, width=2, height=max(1, q1_y - q3_y), color=color)
        _draw_rect(canvas, x=x - half_w, y=med_y - 1, width=2 * half_w, height=3, color=color)
    for center, group in zip(group_centers, groups):
        label = group.upper()
        _draw_bitmap_text(
            canvas,
            text=label,
            x=int(round(center)) - _bitmap_text_width(label, scale=2) // 2,
            y=plot_top + plot_height + 18,
            scale=2,
            color=(30, 30, 30),
        )
    legend_x = width - 245
    legend_y = 38
    legend_items = [
        ("SPARSE DRY FOREST", (194, 118, 44)),
        ("DENSE WET FOREST", (38, 132, 78)),
    ]
    for idx, (label, color) in enumerate(legend_items):
        y = legend_y + idx * 24
        _draw_rect(canvas, x=legend_x, y=y, width=14, height=14, color=color)
        _draw_bitmap_text(canvas, text=label, x=legend_x + 22, y=y + 1, scale=2, color=(40, 40, 40))
    _write_png(path, canvas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moving-image", default=DEFAULT_MOVING_IMAGE)
    parser.add_argument("--fixed-image", default=DEFAULT_FIXED_IMAGE)
    parser.add_argument("--gcp-image", default=DEFAULT_GCP_IMAGE)
    parser.add_argument("--moving-crowns", default=DEFAULT_MOVING_CROWNS)
    parser.add_argument("--fixed-crowns", default=DEFAULT_FIXED_CROWNS)
    parser.add_argument("--dense-moving-image", default=DEFAULT_DENSE_MOVING_IMAGE)
    parser.add_argument("--dense-fixed-image", default=DEFAULT_DENSE_FIXED_IMAGE)
    parser.add_argument("--dense-gcp-image", default=DEFAULT_DENSE_GCP_IMAGE)
    parser.add_argument("--dense-moving-crowns", default=DEFAULT_DENSE_MOVING_CROWNS)
    parser.add_argument("--dense-fixed-crowns", default=DEFAULT_DENSE_FIXED_CROWNS)
    parser.add_argument("--moving-band-index", type=int, default=5)
    parser.add_argument("--dense-moving-band-index", type=int, default=4)
    parser.add_argument("--output-svg", default=DEFAULT_OUTPUT_SVG)
    parser.add_argument("--output-preview", default=DEFAULT_OUTPUT_PREVIEW)
    args = parser.parse_args()

    dry_series = _compute_method_rmse_series(
        moving_image_path=args.moving_image,
        fixed_image_path=args.fixed_image,
        gcp_image_path=args.gcp_image,
        moving_crowns_path=args.moving_crowns,
        fixed_crowns_path=args.fixed_crowns,
        moving_band_index=args.moving_band_index,
    )
    dense_series = _compute_method_rmse_series(
        moving_image_path=args.dense_moving_image,
        fixed_image_path=args.dense_fixed_image,
        gcp_image_path=args.dense_gcp_image,
        moving_crowns_path=args.dense_moving_crowns,
        fixed_crowns_path=args.dense_fixed_crowns,
        moving_band_index=args.dense_moving_band_index,
    )
    color_by_group_site = {
        ("Default RPC", "Dry Forest"): (194, 118, 44),
        ("Default RPC", "Wet Forest"): (38, 132, 78),
        ("Manual GCP", "Dry Forest"): (194, 118, 44),
        ("Manual GCP", "Wet Forest"): (38, 132, 78),
        ("Gradient-Magnitude", "Dry Forest"): (194, 118, 44),
        ("Gradient-Magnitude", "Wet Forest"): (38, 132, 78),
    }
    series_list: list[MethodSeries] = []
    for dry, dense in zip(dry_series, dense_series):
        series_list.append(
            MethodSeries(
                "Dry Forest",
                dry.group,
                color_by_group_site[(dry.group, "Dry Forest")],
                dry.values,
                dry.crown_ids,
            )
        )
        series_list.append(
            MethodSeries(
                "Wet Forest",
                dense.group,
                color_by_group_site[(dense.group, "Wet Forest")],
                dense.values,
                dense.crown_ids,
            )
        )
    _svg_boxplot(args.output_svg, series_list)
    _preview_boxplot(args.output_preview, series_list)
    print(args.output_svg)
    print(args.output_preview)
    for series in series_list:
        print(f"{series.group} {series.label}: median={np.median(series.values):.3f} m, n={len(series.values)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
