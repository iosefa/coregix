from __future__ import annotations

import json
import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.transform import from_origin

from coregix.pipelines import alignment
from coregix.pipelines import alignment_large_main


def test_sample_bilinear_interpolates_and_marks_out_of_bounds() -> None:
    data = np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32)
    rows = np.array([0.5, -1.0], dtype=np.float32)
    cols = np.array([0.5, 0.0], dtype=np.float32)

    sampled, valid = alignment._sample_bilinear(data, rows, cols, fill_value=-9999.0)

    assert sampled.tolist() == [15.0, -9999.0]
    assert valid.tolist() == [True, False]


def test_edge_proxy_uses_only_valid_neighbors() -> None:
    data = np.tile(np.arange(5, dtype=np.float32), (5, 1))
    valid = np.ones((5, 5), dtype=bool)
    valid[:, 0] = False

    edge = alignment._edge_proxy(data, valid)

    assert np.all(edge[:, 0] == 0.0)
    assert np.all(edge[:, 1] == 0.0)
    assert np.all(edge[:, 2:4] > 0.0)


def test_resolve_solve_grid_coarsens_to_requested_resolution() -> None:
    transform = Affine.translation(100.0, 200.0) * Affine.scale(1.0, -1.0)

    width, height, solve_transform = alignment._resolve_solve_grid(
        base_transform=transform,
        base_width=10,
        base_height=8,
        solve_resolution=2.5,
    )

    assert (width, height) == (4, 4)
    assert solve_transform.a == pytest.approx(2.5)
    assert solve_transform.e == pytest.approx(-2.0)
    assert solve_transform.c == pytest.approx(transform.c)
    assert solve_transform.f == pytest.approx(transform.f)


@pytest.mark.parametrize(
    ("split_factor", "width", "height", "expected"),
    [
        (0, 100, 50, (1, 1)),
        (1, 100, 50, (1, 2)),
        (1, 50, 100, (2, 1)),
        (2, 100, 50, (2, 2)),
        (3, 100, 50, (2, 4)),
    ],
)
def test_chunk_grid_shape_prefers_the_long_axis(
    split_factor: int,
    width: int,
    height: int,
    expected: tuple[int, int],
) -> None:
    assert alignment_large_main._chunk_grid_shape(split_factor, width, height) == expected


def test_split_positions_cover_full_extent() -> None:
    assert alignment_large_main._split_positions(10, 3) == [0, 3, 6, 10]


def test_fit_global_rigid_transform_recovers_translation() -> None:
    target = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
        ],
        dtype=np.float64,
    )
    source = target + np.array([2.5, -1.5], dtype=np.float64)

    transform = alignment_large_main._fit_global_rigid_transform(target, source)

    np.testing.assert_allclose(transform.rotation, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(transform.translation, [2.5, -1.5], atol=1e-12)


def test_chunk_grid_shape_rejects_negative_split_factor() -> None:
    with pytest.raises(ValueError, match="split_factor"):
        alignment_large_main._chunk_grid_shape(-1, 10, 10)


def test_multi_pass_solve_resolutions_allow_split_factor_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_large_align(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return alignment.AlignmentResult(output_image_path="out.tif", temp_dir=None)

    monkeypatch.setattr(alignment_large_main, "align_image_pair", fake_large_align)

    result = alignment.align_image_pair(
        "moving.tif",
        "fixed.tif",
        "out.tif",
        split_factor=0,
        solve_resolutions=[6.0, 2.0],
    )

    assert result.output_image_path == "out.tif"
    assert captured["kwargs"]["split_factor"] == 0
    assert captured["kwargs"]["solve_resolution"] is None
    assert captured["kwargs"]["solve_resolutions"] == [6.0, 2.0]


def test_single_pass_coarse_solve_writes_from_original_pixels(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, cols = np.indices((32, 32))
    data = ((rows * 17 + cols * 31) % 1000).astype("int16")
    profile = {
        "driver": "GTiff",
        "height": 32,
        "width": 32,
        "count": 1,
        "dtype": "int16",
        "crs": "EPSG:32605",
        "transform": from_origin(500000.0, 1000.0, 1.0, 1.0),
        "nodata": -9999,
    }
    fixed_path = tmp_path / "fixed.tif"
    moving_path = tmp_path / "moving.tif"
    output_path = tmp_path / "aligned.tif"
    for image_path in (fixed_path, moving_path):
        with rasterio.open(image_path, "w", **profile) as dst:
            dst.write(data, 1)

    def fake_estimate_elastix_transform(**kwargs):
        return object()

    def fake_deformation_field_from_transform(
        fixed_image_path, transform_parameter_object, output_directory
    ):
        with rasterio.open(fixed_image_path) as src:
            return np.zeros((src.height, src.width, 2), dtype=np.float32)

    monkeypatch.setattr(
        alignment, "estimate_elastix_transform", fake_estimate_elastix_transform
    )
    monkeypatch.setattr(
        alignment, "deformation_field_from_transform", fake_deformation_field_from_transform
    )

    alignment.align_image_pair(
        moving_image_path=str(moving_path),
        fixed_image_path=str(fixed_path),
        output_image_path=str(output_path),
        moving_band_index=0,
        fixed_band_index=0,
        output_on_moving_grid=False,
        split_factor=0,
        solve_resolutions=[8.0],
        use_edge_proxies=False,
    )

    with rasterio.open(output_path) as src:
        result = src.read(1)
        assert src.res == (1.0, 1.0)

    np.testing.assert_array_equal(result, data)


def test_single_pass_writes_optional_transform_json(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows, cols = np.indices((16, 16))
    data = (rows + cols).astype("int16")
    profile = {
        "driver": "GTiff",
        "height": 16,
        "width": 16,
        "count": 1,
        "dtype": "int16",
        "crs": "EPSG:32605",
        "transform": from_origin(500000.0, 1000.0, 1.0, 1.0),
        "nodata": -9999,
    }
    fixed_path = tmp_path / "fixed.tif"
    moving_path = tmp_path / "moving.tif"
    output_path = tmp_path / "aligned.tif"
    transform_json_path = tmp_path / "aligned.coregix.json"
    for image_path in (fixed_path, moving_path):
        with rasterio.open(image_path, "w", **profile) as dst:
            dst.write(data, 1)

    class FakeParameterObject:
        def GetNumberOfParameterMaps(self):
            return 1

        def GetParameterMap(self, idx):
            return {
                "Transform": ["TranslationTransform"],
                "TransformParameters": ["0", "0"],
            }

    def fake_estimate_elastix_transform(**kwargs):
        return FakeParameterObject()

    def fake_deformation_field_from_transform(
        fixed_image_path, transform_parameter_object, output_directory
    ):
        with rasterio.open(fixed_image_path) as src:
            return np.zeros((src.height, src.width, 2), dtype=np.float32)

    monkeypatch.setattr(
        alignment, "estimate_elastix_transform", fake_estimate_elastix_transform
    )
    monkeypatch.setattr(
        alignment, "deformation_field_from_transform", fake_deformation_field_from_transform
    )

    result = alignment.align_image_pair(
        moving_image_path=str(moving_path),
        fixed_image_path=str(fixed_path),
        output_image_path=str(output_path),
        moving_band_index=0,
        fixed_band_index=0,
        output_on_moving_grid=False,
        output_transform_json_path=str(transform_json_path),
        split_factor=0,
        solve_resolutions=[4.0],
        use_edge_proxies=False,
    )

    assert result.output_transform_json_path == str(transform_json_path)
    metadata = json.loads(transform_json_path.read_text())
    assert metadata["coregix_transform_schema_version"] == 1
    assert metadata["transform_model"] == "elastix_translation_rigid_world_affine"
    assert metadata["paths"]["output_image"] == str(output_path)
    np.testing.assert_allclose(
        metadata["source_to_target"]["matrix"],
        np.eye(3),
        atol=1e-9,
    )
