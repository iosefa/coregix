"""Generic vector alignment evaluation utilities."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from affine import Affine
from rasterio.features import rasterize


@dataclass
class FeatureAlignmentError:
    feature_id: str
    initial_rmse_m: float
    aligned_rmse_m: float
    rmse_m: float


@dataclass
class VectorAlignmentResult:
    rmse_m: float
    initial_rmse_m: float
    aligned_rmse_m: float
    improvement_m: float
    improvement_percent: Optional[float]
    feature_count: int
    per_feature: list[FeatureAlignmentError]
    output_json_path: Optional[str] = None
    output_csv_path: Optional[str] = None


def _load_geometries_by_id(path: str, *, id_field: str) -> dict[str, dict[str, Any]]:
    try:
        from osgeo import ogr
    except ImportError as exc:
        raise RuntimeError(
            "GDAL/OGR is required for vector alignment evaluation."
        ) from exc

    ogr.DontUseExceptions()
    ds = ogr.Open(path)
    if ds is None:
        raise RuntimeError(f"Failed to open vector file: {path}")
    layer = ds.GetLayer(0)
    if layer is None:
        raise RuntimeError(f"No vector layer found in {path}")

    out: dict[str, dict[str, Any]] = {}
    for feat in layer:
        feature_id = feat.GetField(id_field)
        geom = feat.GetGeometryRef()
        if feature_id is None or geom is None:
            continue
        out[str(feature_id)] = json.loads(geom.ExportToJson())
    ds = None
    if not out:
        raise RuntimeError(f"No geometries with id field '{id_field}' found in {path}")
    return out


def _transform_xy(matrix: np.ndarray, x: float, y: float) -> list[float]:
    mapped = matrix @ np.array([float(x), float(y), 1.0], dtype=np.float64)
    return [float(mapped[0]), float(mapped[1])]


def _transform_coordinates(obj: Any, matrix: np.ndarray) -> Any:
    if isinstance(obj, list):
        if len(obj) >= 2 and all(isinstance(v, (int, float)) for v in obj[:2]):
            xy = _transform_xy(matrix, float(obj[0]), float(obj[1]))
            return xy + obj[2:]
        return [_transform_coordinates(item, matrix) for item in obj]
    return obj


def _transform_geometry(geometry: dict[str, Any], matrix: np.ndarray) -> dict[str, Any]:
    geom_type = geometry.get("type")
    if geom_type == "GeometryCollection":
        return {
            **geometry,
            "geometries": [
                _transform_geometry(child, matrix)
                for child in geometry.get("geometries", [])
            ],
        }
    return {
        **geometry,
        "coordinates": _transform_coordinates(geometry["coordinates"], matrix),
    }


def _geometry_bounds(geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    coords: list[tuple[float, float]] = []

    def visit(obj: Any) -> None:
        if isinstance(obj, list):
            if len(obj) >= 2 and all(isinstance(v, (int, float)) for v in obj[:2]):
                coords.append((float(obj[0]), float(obj[1])))
            else:
                for item in obj:
                    visit(item)

    if geometry.get("type") == "GeometryCollection":
        for child in geometry.get("geometries", []):
            minx, miny, maxx, maxy = _geometry_bounds(child)
            coords.extend([(minx, miny), (maxx, maxy)])
    else:
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


def _rasterize_pair(
    fixed_geom: dict[str, Any],
    moving_geom: dict[str, Any],
    *,
    pixel_size: float,
    padding: float,
) -> tuple[np.ndarray, np.ndarray]:
    bounds = [_geometry_bounds(fixed_geom), _geometry_bounds(moving_geom)]
    minx = min(b[0] for b in bounds) - padding
    miny = min(b[1] for b in bounds) - padding
    maxx = max(b[2] for b in bounds) + padding
    maxy = max(b[3] for b in bounds) + padding
    width = max(1, int(math.ceil((maxx - minx) / pixel_size)))
    height = max(1, int(math.ceil((maxy - miny) / pixel_size)))
    transform = Affine(pixel_size, 0.0, minx, 0.0, -pixel_size, maxy)
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


def _write_json(path: str, result: VectorAlignmentResult) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rmse_m": result.rmse_m,
        "initial_rmse_m": result.initial_rmse_m,
        "aligned_rmse_m": result.aligned_rmse_m,
        "improvement_m": result.improvement_m,
        "improvement_percent": result.improvement_percent,
        "feature_count": result.feature_count,
        "per_feature": [
            {
                "feature_id": item.feature_id,
                "initial_rmse_m": item.initial_rmse_m,
                "aligned_rmse_m": item.aligned_rmse_m,
                "rmse_m": item.rmse_m,
            }
            for item in result.per_feature
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: str, result: VectorAlignmentResult) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "feature_id",
                "initial_rmse_m",
                "aligned_rmse_m",
                "rmse_m",
                "improvement_m",
                "improvement_percent",
            ],
        )
        writer.writeheader()
        for item in result.per_feature:
            improvement_m = item.initial_rmse_m - item.aligned_rmse_m
            improvement_percent = (
                None
                if item.initial_rmse_m == 0.0
                else 100.0 * improvement_m / item.initial_rmse_m
            )
            writer.writerow(
                {
                    "feature_id": item.feature_id,
                    "initial_rmse_m": f"{item.initial_rmse_m:.6f}",
                    "aligned_rmse_m": f"{item.aligned_rmse_m:.6f}",
                    "rmse_m": f"{item.rmse_m:.6f}",
                    "improvement_m": f"{improvement_m:.6f}",
                    "improvement_percent": ""
                    if improvement_percent is None
                    else f"{improvement_percent:.6f}",
                }
            )


def evaluate_vector_alignment(
    *,
    fixed_vector_path: str,
    moving_vector_path: str,
    transform_json_path: str,
    id_field: str,
    pixel_size: float = 0.5,
    padding: float = 8.0,
    output_json_path: Optional[str] = None,
    output_csv_path: Optional[str] = None,
) -> VectorAlignmentResult:
    """Evaluate alignment of paired vector geometries before and after a transform.

    The transform JSON must be produced by Coregix and contain a
    ``source_to_target`` matrix. Paired geometries sharing ``id_field`` are
    compared with symmetric boundary RMSE before and after that matrix is
    applied to the moving geometries.
    """
    if pixel_size <= 0:
        raise ValueError("pixel_size must be > 0.")
    if padding < 0:
        raise ValueError("padding must be >= 0.")

    transform_metadata = json.loads(Path(transform_json_path).read_text(encoding="utf-8"))
    matrix = np.asarray(transform_metadata["source_to_target"]["matrix"], dtype=np.float64)
    fixed_geoms = _load_geometries_by_id(fixed_vector_path, id_field=id_field)
    moving_geoms = _load_geometries_by_id(moving_vector_path, id_field=id_field)
    common_ids = sorted(set(fixed_geoms) & set(moving_geoms))
    if not common_ids:
        raise ValueError("No shared feature ids found between fixed and moving vectors.")

    per_feature: list[FeatureAlignmentError] = []
    for feature_id in common_ids:
        initial_fixed_mask, initial_moving_mask = _rasterize_pair(
            fixed_geoms[feature_id],
            moving_geoms[feature_id],
            pixel_size=pixel_size,
            padding=padding,
        )
        initial_rmse = (
            _symmetric_boundary_rmse(initial_fixed_mask, initial_moving_mask)
            * pixel_size
        )

        transformed_moving = _transform_geometry(moving_geoms[feature_id], matrix)
        aligned_fixed_mask, aligned_moving_mask = _rasterize_pair(
            fixed_geoms[feature_id],
            transformed_moving,
            pixel_size=pixel_size,
            padding=padding,
        )
        aligned_rmse = (
            _symmetric_boundary_rmse(aligned_fixed_mask, aligned_moving_mask)
            * pixel_size
        )
        per_feature.append(
            FeatureAlignmentError(
                feature_id=feature_id,
                initial_rmse_m=initial_rmse,
                aligned_rmse_m=aligned_rmse,
                rmse_m=aligned_rmse,
            )
        )

    initial_rmse_m = float(
        np.sqrt(np.mean([item.initial_rmse_m**2 for item in per_feature]))
    )
    aligned_rmse_m = float(
        np.sqrt(np.mean([item.aligned_rmse_m**2 for item in per_feature]))
    )
    improvement_m = initial_rmse_m - aligned_rmse_m
    improvement_percent = (
        None
        if initial_rmse_m == 0.0
        else 100.0 * improvement_m / initial_rmse_m
    )
    result = VectorAlignmentResult(
        rmse_m=aligned_rmse_m,
        initial_rmse_m=initial_rmse_m,
        aligned_rmse_m=aligned_rmse_m,
        improvement_m=improvement_m,
        improvement_percent=improvement_percent,
        feature_count=len(per_feature),
        per_feature=per_feature,
        output_json_path=output_json_path,
        output_csv_path=output_csv_path,
    )
    if output_json_path is not None:
        _write_json(output_json_path, result)
    if output_csv_path is not None:
        _write_csv(output_csv_path, result)
    return result


__all__ = [
    "FeatureAlignmentError",
    "VectorAlignmentResult",
    "evaluate_vector_alignment",
]
