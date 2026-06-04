from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from coregix.pipelines.apply_transform import apply_coregix_transform
from tests.helpers import DEFAULT_CRS, write_test_raster


def _affine_list(transform):
    return [
        float(transform.a),
        float(transform.b),
        float(transform.c),
        float(transform.d),
        float(transform.e),
        float(transform.f),
    ]


def _write_transform_json(
    path,
    *,
    moving_path,
    moving_width,
    moving_height,
    moving_transform,
    output_width,
    output_height,
    output_transform,
    target_to_source,
    output_on_moving_grid=False,
    moving_window=None,
):
    if moving_window is None:
        moving_window = {
            "col_off": 0.0,
            "row_off": 0.0,
            "width": float(moving_width),
            "height": float(moving_height),
        }
    metadata = {
        "coregix_transform_schema_version": 1,
        "paths": {"moving_image": str(moving_path), "fixed_image": "fixed.tif"},
        "crs": {"moving": DEFAULT_CRS, "fixed": DEFAULT_CRS},
        "options": {
            "output_on_moving_grid": output_on_moving_grid,
            "output_nodata": -9999.0,
        },
        "rasters": {
            "moving": {
                "width": moving_width,
                "height": moving_height,
                "transform": _affine_list(moving_transform),
            },
            "output": {
                "width": output_width,
                "height": output_height,
                "transform": _affine_list(output_transform),
            },
        },
        "windows": {"moving": moving_window},
        "target_to_source": {
            "type": "affine_2d_homogeneous_matrix",
            "matrix": target_to_source,
        },
    }
    path.write_text(json.dumps(metadata), encoding="utf-8")


def test_apply_coregix_transform_identity_reference_grid(tmp_path) -> None:
    transform = from_origin(0.0, 5.0, 1.0, 1.0)
    data = np.arange(25, dtype=np.int16).reshape(5, 5)
    moving_path = write_test_raster(tmp_path / "moving.tif", data, transform=transform)
    transform_json = tmp_path / "transform.json"
    output_path = tmp_path / "aligned.tif"
    _write_transform_json(
        transform_json,
        moving_path=moving_path,
        moving_width=5,
        moving_height=5,
        moving_transform=transform,
        output_width=5,
        output_height=5,
        output_transform=transform,
        target_to_source=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )

    result = apply_coregix_transform(
        moving_image_path=str(moving_path),
        transform_json_path=str(transform_json),
        output_image_path=str(output_path),
    )

    assert result.output_image_path == str(output_path)
    with rasterio.open(output_path) as src:
        assert src.width == 5
        assert src.height == 5
        assert src.transform == transform
        assert src.crs.to_string() == DEFAULT_CRS
        assert src.nodata == -9999
        np.testing.assert_array_equal(src.read(1), data)


def test_apply_coregix_transform_samples_with_target_to_source_matrix(tmp_path) -> None:
    transform = from_origin(0.0, 5.0, 1.0, 1.0)
    data = np.arange(25, dtype=np.int16).reshape(5, 5)
    moving_path = write_test_raster(tmp_path / "moving.tif", data, transform=transform)
    transform_json = tmp_path / "transform.json"
    output_path = tmp_path / "shifted.tif"
    _write_transform_json(
        transform_json,
        moving_path=moving_path,
        moving_width=5,
        moving_height=5,
        moving_transform=transform,
        output_width=5,
        output_height=5,
        output_transform=transform,
        target_to_source=[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )

    apply_coregix_transform(
        moving_image_path=str(moving_path),
        transform_json_path=str(transform_json),
        output_image_path=str(output_path),
    )

    expected = np.full((5, 5), -9999, dtype=np.int16)
    expected[:, :4] = data[:, 1:]
    with rasterio.open(output_path) as src:
        np.testing.assert_array_equal(src.read(1), expected)


def test_apply_coregix_transform_preserves_source_pixels_outside_moving_window(tmp_path) -> None:
    transform = from_origin(0.0, 5.0, 1.0, 1.0)
    data = np.arange(25, dtype=np.int16).reshape(5, 5)
    moving_path = write_test_raster(tmp_path / "moving.tif", data, transform=transform)
    transform_json = tmp_path / "transform.json"
    output_path = tmp_path / "moving_grid.tif"
    _write_transform_json(
        transform_json,
        moving_path=moving_path,
        moving_width=5,
        moving_height=5,
        moving_transform=transform,
        output_width=5,
        output_height=5,
        output_transform=transform,
        output_on_moving_grid=True,
        moving_window={"col_off": 1.0, "row_off": 1.0, "width": 3.0, "height": 3.0},
        target_to_source=[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )

    apply_coregix_transform(
        moving_image_path=str(moving_path),
        transform_json_path=str(transform_json),
        output_image_path=str(output_path),
    )

    with rasterio.open(output_path) as src:
        out = src.read(1)
    np.testing.assert_array_equal(out[0, :], data[0, :])
    np.testing.assert_array_equal(out[:, 0], data[:, 0])
    np.testing.assert_array_equal(out[1:4, 1:4], data[1:4, 2:5])


def test_apply_coregix_transform_rejects_moving_raster_mismatch(tmp_path) -> None:
    transform = from_origin(0.0, 5.0, 1.0, 1.0)
    moving_path = write_test_raster(tmp_path / "moving.tif", np.ones((5, 5), dtype=np.int16), transform=transform)
    transform_json = tmp_path / "transform.json"
    _write_transform_json(
        transform_json,
        moving_path=tmp_path / "other.tif",
        moving_width=5,
        moving_height=5,
        moving_transform=transform,
        output_width=5,
        output_height=5,
        output_transform=transform,
        target_to_source=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )

    with pytest.raises(ValueError, match="Moving image path"):
        apply_coregix_transform(
            moving_image_path=str(moving_path),
            transform_json_path=str(transform_json),
            output_image_path=str(tmp_path / "out.tif"),
        )
