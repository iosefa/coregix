#!/usr/bin/env python3
"""Compute boundary RMSE for group 7 alignment test crown GeoPackages."""

from __future__ import annotations

import csv
import html
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from affine import Affine
from osgeo import ogr
from rasterio.features import rasterize


TEST_DIR = Path(os.path.expanduser("~/Projects/DryForest/alignment_tests_grp7"))
FIXED_PATH = TEST_DIR / "fixed_grp7_crown1.gpkg"
OUT_CSV = TEST_DIR / "alignment_rmse_summary.csv"
OUT_SVG = TEST_DIR / "alignment_rmse_bar_chart.svg"
PIXEL_SIZE_M = 0.5
PADDING_M = 8.0


@dataclass
class Result:
    file: str
    label: str
    rmse_m: float
    crown_count: int
    band: str
    edge_mode: str
    split_factor: str
    solve_resolutions: str


def _load_polygons_by_id(path: Path, *, id_field: str = "crown_id") -> dict[int, dict]:
    ogr.DontUseExceptions()
    ds = ogr.Open(str(path))
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


def _geometry_bounds(geometry: dict) -> tuple[float, float, float, float]:
    coords: list[tuple[float, float]] = []

    def visit(obj: object) -> None:
        if isinstance(obj, list):
            if len(obj) >= 2 and all(isinstance(v, (int, float)) for v in obj[:2]):
                coords.append((float(obj[0]), float(obj[1])))
            else:
                for item in obj:
                    visit(item)

    visit(geometry["coordinates"])
    if not coords:
        raise ValueError("Geometry has no coordinates")
    xs = [xy[0] for xy in coords]
    ys = [xy[1] for xy in coords]
    return min(xs), min(ys), max(xs), max(ys)


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


def _boundary(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    eroded = _binary_erosion(mask, iterations=1)
    return _binary_dilation(mask & ~eroded, iterations=1)


def _symmetric_boundary_rmse(a: np.ndarray, b: np.ndarray) -> float:
    a_pts = np.argwhere(_boundary(a))
    b_pts = np.argwhere(_boundary(b))
    if a_pts.size == 0 or b_pts.size == 0:
        return float("inf")
    dists = np.sqrt(((a_pts[:, None, :] - b_pts[None, :, :]) ** 2).sum(axis=2))
    a_to_b = dists.min(axis=1)
    b_to_a = dists.min(axis=0)
    return float(np.sqrt((np.mean(a_to_b**2) + np.mean(b_to_a**2)) / 2.0))


def _rasterize_pair(fixed_geom: dict, moving_geom: dict) -> tuple[np.ndarray, np.ndarray]:
    bounds = [_geometry_bounds(fixed_geom), _geometry_bounds(moving_geom)]
    minx = min(b[0] for b in bounds) - PADDING_M
    miny = min(b[1] for b in bounds) - PADDING_M
    maxx = max(b[2] for b in bounds) + PADDING_M
    maxy = max(b[3] for b in bounds) + PADDING_M
    width = max(1, int(math.ceil((maxx - minx) / PIXEL_SIZE_M)))
    height = max(1, int(math.ceil((maxy - miny) / PIXEL_SIZE_M)))
    transform = Affine(PIXEL_SIZE_M, 0.0, minx, 0.0, -PIXEL_SIZE_M, maxy)
    fixed = rasterize(
        [(fixed_geom, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)
    moving = rasterize(
        [(moving_geom, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)
    return fixed, moving


def _parse_label(path: Path) -> tuple[str, str, str, str, str]:
    stem = path.stem
    band_match = re.search(r"_band_(\d+)_", stem)
    split_match = re.search(r"_split_(\d+)_", stem)
    solve_match = re.search(r"_solve_([0-9_p]+)_aligned$", stem)
    if "_raw_" in stem:
        edge_mode = "raw"
    elif "_edge_" in stem:
        edge_mode = "edge"
    else:
        edge_mode = "default"
    band = band_match.group(1) if band_match else "default"
    split = split_match.group(1) if split_match else "default"
    solve = solve_match.group(1).replace("_", ",").replace("p", ".") if solve_match else "unknown"
    label = f"band {band}, {edge_mode}, split {split}, {solve}"
    return label, band, edge_mode, split, solve


def _compute_results() -> list[Result]:
    fixed_polys = _load_polygons_by_id(FIXED_PATH)
    results: list[Result] = []
    for path in sorted(TEST_DIR.glob("*.gpkg")):
        if path.name == FIXED_PATH.name:
            continue
        moving_polys = _load_polygons_by_id(path)
        common_ids = sorted(set(fixed_polys) & set(moving_polys))
        if not common_ids:
            continue
        vals: list[float] = []
        for crown_id in common_ids:
            fixed_mask, moving_mask = _rasterize_pair(fixed_polys[crown_id], moving_polys[crown_id])
            vals.append(_symmetric_boundary_rmse(fixed_mask, moving_mask) * PIXEL_SIZE_M)
        label, band, edge_mode, split, solve = _parse_label(path)
        results.append(
            Result(
                file=path.name,
                label=label,
                rmse_m=float(np.sqrt(np.mean(np.square(vals)))),
                crown_count=len(vals),
                band=band,
                edge_mode=edge_mode,
                split_factor=split,
                solve_resolutions=solve,
            )
        )
    return sorted(results, key=lambda r: r.rmse_m)


def _write_csv(results: list[Result]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "rmse_m",
                "crown_count",
                "band",
                "edge_mode",
                "split_factor",
                "solve_resolutions",
                "file",
            ],
        )
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "rmse_m": f"{result.rmse_m:.6f}",
                    "crown_count": result.crown_count,
                    "band": result.band,
                    "edge_mode": result.edge_mode,
                    "split_factor": result.split_factor,
                    "solve_resolutions": result.solve_resolutions,
                    "file": result.file,
                }
            )


def _write_svg(results: list[Result]) -> None:
    width = 1120
    height = 660
    left = 88
    right = 28
    top = 36
    bottom = 220
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_rmse = max(r.rmse_m for r in results)
    y_max = max(1.0, math.ceil(max_rmse * 2.0) / 2.0)
    bar_gap = 7
    bar_w = max(8, (plot_w - bar_gap * (len(results) - 1)) / len(results))

    def y_for(v: float) -> float:
        return top + plot_h - (v / y_max) * plot_h

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="18" font-family="Helvetica, Arial, sans-serif" font-weight="700">Group 7 Crown Alignment RMSE</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222" stroke-width="1.2"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222" stroke-width="1.2"/>',
        f'<text x="24" y="{top + plot_h / 2}" transform="rotate(-90 24 {top + plot_h / 2})" text-anchor="middle" font-size="15" font-family="Helvetica, Arial, sans-serif">Boundary RMSE (m)</text>',
    ]
    tick = 0.0
    while tick <= y_max + 1e-9:
        y = y_for(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e5e5" stroke-width="1"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="Helvetica, Arial, sans-serif">{tick:.1f}</text>')
        tick += 0.5

    for i, result in enumerate(results):
        x = left + i * (bar_w + bar_gap)
        y = y_for(result.rmse_m)
        h = top + plot_h - y
        color = {"edge": "#2b83ba", "raw": "#d95f02"}.get(result.edge_mode, "#4d4d4d")
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w / 2:.2f}" y="{y - 5:.2f}" text-anchor="middle" font-size="11" font-family="Helvetica, Arial, sans-serif">{result.rmse_m:.2f}</text>')
        label = html.escape(result.label)
        parts.append(f'<text x="{x + bar_w / 2:.2f}" y="{top + plot_h + 16}" text-anchor="end" font-size="11" font-family="Helvetica, Arial, sans-serif" transform="rotate(-55 {x + bar_w / 2:.2f} {top + plot_h + 16})">{label}</text>')

    parts.append(f'<text x="{left}" y="{height - 24}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="#555">Gray = default CLI edge behavior; blue = explicit edge proxy; orange = raw intensity. Metric matches manuscript symmetric boundary RMSE at 0.5 m pixels.</text>')
    parts.append("</svg>")
    OUT_SVG.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    results = _compute_results()
    if not results:
        raise RuntimeError(f"No result GeoPackages found in {TEST_DIR}")
    _write_csv(results)
    _write_svg(results)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_SVG}")
    print("Top results:")
    for result in results[:8]:
        print(f"{result.rmse_m:.3f} m | {result.label} | n={result.crown_count}")


if __name__ == "__main__":
    main()
