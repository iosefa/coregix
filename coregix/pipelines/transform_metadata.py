"""Transform metadata helpers for alignment outputs."""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Sequence

import numpy as np
from affine import Affine
from rasterio.windows import Window


SCHEMA_VERSION = 1


def affine_to_list(transform: Affine) -> list[float]:
    """Return affine coefficients in rasterio/Affine order."""
    return [
        float(transform.a),
        float(transform.b),
        float(transform.c),
        float(transform.d),
        float(transform.e),
        float(transform.f),
    ]


def window_to_dict(window: Window) -> dict[str, float]:
    return {
        "col_off": float(window.col_off),
        "row_off": float(window.row_off),
        "width": float(window.width),
        "height": float(window.height),
    }


def matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def invert_affine_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(matrix, dtype=np.float64))


def fit_affine_matrix(target_xy: np.ndarray, source_xy: np.ndarray) -> np.ndarray:
    """Fit a 2D homogeneous affine matrix from target to source coordinates."""
    target_xy = np.asarray(target_xy, dtype=np.float64)
    source_xy = np.asarray(source_xy, dtype=np.float64)
    if target_xy.shape != (3, 2) or source_xy.shape != (3, 2):
        raise ValueError("target_xy and source_xy must both have shape (3, 2).")
    design = np.column_stack(
        [target_xy[:, 0], target_xy[:, 1], np.ones(3, dtype=np.float64)]
    )
    x_coeff = np.linalg.solve(design, source_xy[:, 0])
    y_coeff = np.linalg.solve(design, source_xy[:, 1])
    return np.array(
        [
            [x_coeff[0], x_coeff[1], x_coeff[2]],
            [y_coeff[0], y_coeff[1], y_coeff[2]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def matrix_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    return np.array(
        [
            [rotation[0, 0], rotation[0, 1], translation[0]],
            [rotation[1, 0], rotation[1, 1], translation[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def transform_points(matrix: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    homogeneous = np.column_stack(
        [points_xy[:, 0], points_xy[:, 1], np.ones(points_xy.shape[0], dtype=np.float64)]
    )
    mapped = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    return mapped[:, :2]


def crs_to_string(crs: Any) -> Optional[str]:
    return None if crs is None else str(crs)


def solve_resolutions_to_list(
    solve_resolution: Optional[float],
    solve_resolutions: Optional[Sequence[Optional[float]]],
) -> list[Optional[float]]:
    if solve_resolutions is not None:
        return [None if value is None else float(value) for value in solve_resolutions]
    return [None if solve_resolution is None else float(solve_resolution)]


def parameter_object_to_list(transform_parameter_object: Any) -> list[dict[str, list[str]]]:
    """Serialize an elastix parameter object without depending on ITK JSON support."""
    maps: list[dict[str, list[str]]] = []
    if transform_parameter_object is None:
        return maps
    for idx in range(int(transform_parameter_object.GetNumberOfParameterMaps())):
        parameter_map = transform_parameter_object.GetParameterMap(idx)
        serialized: dict[str, list[str]] = {}
        for key in parameter_map.keys():
            serialized[str(key)] = [str(value) for value in parameter_map[key]]
        maps.append(serialized)
    return maps


def base_transform_metadata(
    *,
    moving_image_path: str,
    fixed_image_path: str,
    output_image_path: Optional[str],
    output_transform_json_path: str,
    moving_band_index: int,
    fixed_band_index: int,
    split_factor: int,
    solve_resolution: Optional[float],
    solve_resolutions: Optional[Sequence[Optional[float]]],
    output_on_moving_grid: bool,
    clip_fixed_to_moving: bool,
    enforce_mutual_valid_mask: bool,
    use_edge_proxies: bool,
    moving_nodata: Optional[float],
    fixed_nodata: Optional[float],
    output_nodata: float,
    fixed_crs: Any,
    moving_crs: Any,
    fixed_width: int,
    fixed_height: int,
    fixed_transform: Affine,
    moving_width: int,
    moving_height: int,
    moving_transform: Affine,
    output_width: int,
    output_height: int,
    output_transform: Affine,
    fixed_window: Window,
    moving_window: Window,
    fixed_window_transform: Affine,
    moving_window_transform: Affine,
    solve_width: int,
    solve_height: int,
    solve_transform: Affine,
) -> dict[str, Any]:
    return {
        "coregix_transform_schema_version": SCHEMA_VERSION,
        "paths": {
            "moving_image": moving_image_path,
            "fixed_image": fixed_image_path,
            "output_image": output_image_path,
            "output_transform_json": output_transform_json_path,
        },
        "coordinate_convention": {
            "target_to_source": (
                "Maps target/output-reference CRS x,y coordinates to original moving/source "
                "CRS x,y coordinates sampled by Coregix."
            ),
            "source_to_target": (
                "Inverse of target_to_source. Use this to transform original moving "
                "vector geometries into the fixed/aligned coordinate frame."
            ),
            "units": "raster CRS units",
        },
        "crs": {
            "fixed": crs_to_string(fixed_crs),
            "moving": crs_to_string(moving_crs),
        },
        "bands": {
            "moving_band_index": int(moving_band_index),
            "fixed_band_index": int(fixed_band_index),
        },
        "options": {
            "split_factor": int(split_factor),
            "solve_resolution": None if solve_resolution is None else float(solve_resolution),
            "solve_resolutions": solve_resolutions_to_list(solve_resolution, solve_resolutions),
            "output_on_moving_grid": bool(output_on_moving_grid),
            "clip_fixed_to_moving": bool(clip_fixed_to_moving),
            "enforce_mutual_valid_mask": bool(enforce_mutual_valid_mask),
            "use_edge_proxies": bool(use_edge_proxies),
            "moving_nodata": moving_nodata,
            "fixed_nodata": fixed_nodata,
            "output_nodata": float(output_nodata),
        },
        "rasters": {
            "fixed": {
                "width": int(fixed_width),
                "height": int(fixed_height),
                "transform": affine_to_list(fixed_transform),
            },
            "moving": {
                "width": int(moving_width),
                "height": int(moving_height),
                "transform": affine_to_list(moving_transform),
            },
            "output": {
                "width": int(output_width),
                "height": int(output_height),
                "transform": affine_to_list(output_transform),
            },
        },
        "windows": {
            "fixed": window_to_dict(fixed_window),
            "moving": window_to_dict(moving_window),
            "fixed_transform": affine_to_list(fixed_window_transform),
            "moving_transform": affine_to_list(moving_window_transform),
        },
        "solve_grid": {
            "width": int(solve_width),
            "height": int(solve_height),
            "transform": affine_to_list(solve_transform),
        },
    }


def add_matrix_metadata(metadata: dict[str, Any], target_to_source_matrix: np.ndarray) -> None:
    target_to_source_matrix = np.asarray(target_to_source_matrix, dtype=np.float64)
    source_to_target_matrix = invert_affine_matrix(target_to_source_matrix)
    metadata["target_to_source"] = {
        "type": "affine_2d_homogeneous_matrix",
        "matrix": matrix_to_list(target_to_source_matrix),
    }
    metadata["source_to_target"] = {
        "type": "affine_2d_homogeneous_matrix",
        "matrix": matrix_to_list(source_to_target_matrix),
    }


def write_transform_metadata(path: str, metadata: dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    return path
