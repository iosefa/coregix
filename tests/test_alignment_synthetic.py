from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from coregix.pipelines import alignment
from tests.helpers import DEFAULT_CRS, write_test_raster


@pytest.fixture
def mocked_identity_registration(monkeypatch: pytest.MonkeyPatch):
    calls = {"estimate": []}

    def fake_estimate_elastix_transform(**kwargs):
        calls["estimate"].append(kwargs)
        return object()

    def fake_deformation_field_from_transform(
        reference_image_path,
        transform_parameter_object,
        *,
        output_directory=None,
    ):
        with rasterio.open(reference_image_path) as src:
            return np.zeros((src.height, src.width, 2), dtype=np.float32)

    monkeypatch.setattr(alignment, "estimate_elastix_transform", fake_estimate_elastix_transform)
    monkeypatch.setattr(alignment, "deformation_field_from_transform", fake_deformation_field_from_transform)
    return calls


def test_align_image_pair_identity_transform_preserves_source_grid(
    tmp_path,
    mocked_identity_registration,
) -> None:
    transform = from_origin(100.0, 200.0, 1.0, 1.0)
    band1 = np.arange(64, dtype=np.int16).reshape(8, 8)
    moving_data = np.stack([band1, band1 + 1000])
    fixed_data = moving_data.copy()
    moving_path = write_test_raster(tmp_path / "source.tif", moving_data, transform=transform)
    fixed_path = write_test_raster(tmp_path / "reference.tif", fixed_data, transform=transform)
    output_path = tmp_path / "aligned.tif"

    result = alignment.align_image_pair(
        moving_image_path=str(moving_path),
        fixed_image_path=str(fixed_path),
        output_image_path=str(output_path),
        temp_dir=str(tmp_path),
        keep_temp_dir=True,
        use_edge_proxies=False,
    )

    assert result.output_image_path == str(output_path)
    assert result.temp_dir is not None
    assert mocked_identity_registration["estimate"][0]["parameter_map"] == ["translation", "rigid"]

    with rasterio.open(output_path) as src:
        assert src.count == 2
        assert src.width == 8
        assert src.height == 8
        assert src.crs.to_string() == DEFAULT_CRS
        assert src.transform == transform
        assert src.nodata == -9999
        np.testing.assert_array_equal(src.read(), moving_data)

    with rasterio.open(f"{result.temp_dir}/fixed_mask.tif") as mask_src:
        assert int(mask_src.read(1).sum()) == 64


def test_align_image_pair_bspline_uses_nonrigid_parameter_maps(
    tmp_path,
    mocked_identity_registration,
) -> None:
    transform = from_origin(100.0, 200.0, 1.0, 1.0)
    data = np.arange(64, dtype=np.int16).reshape(8, 8)
    moving_path = write_test_raster(tmp_path / "source.tif", data, transform=transform)
    fixed_path = write_test_raster(tmp_path / "reference.tif", data, transform=transform)
    output_path = tmp_path / "aligned_bspline.tif"

    result = alignment.align_image_pair(
        moving_image_path=str(moving_path),
        fixed_image_path=str(fixed_path),
        output_image_path=str(output_path),
        temp_dir=str(tmp_path),
        keep_temp_dir=True,
        use_edge_proxies=False,
        transform_model="bspline",
        solve_resolutions=[2.0],
    )

    assert mocked_identity_registration["estimate"][0]["parameter_map"] == [
        "translation",
        "rigid",
        "bspline",
    ]
    assert result.temp_dir is not None
    with rasterio.open(f"{result.temp_dir}/fixed_reg.tif") as src:
        assert (src.width, src.height) == (4, 4)


def test_align_image_pair_rejects_unknown_transform_model() -> None:
    with pytest.raises(ValueError, match="transform_model"):
        alignment.align_image_pair(
            "moving.tif",
            "fixed.tif",
            "out.tif",
            transform_model="affine",
        )


def test_align_image_pair_bspline_rejects_dry_run_before_json_requirement() -> None:
    with pytest.raises(ValueError, match="dry_run is not supported"):
        alignment.align_image_pair(
            "moving.tif",
            "fixed.tif",
            transform_model="bspline",
            dry_run=True,
        )


def test_align_image_pair_bspline_rejects_transform_json(tmp_path) -> None:
    with pytest.raises(ValueError, match="output_transform_json_path"):
        alignment.align_image_pair(
            "moving.tif",
            "fixed.tif",
            "out.tif",
            transform_model="bspline",
            output_transform_json_path=str(tmp_path / "transform.json"),
        )


def test_align_image_pair_bspline_rejects_split_factor() -> None:
    with pytest.raises(ValueError, match="split_factor"):
        alignment.align_image_pair(
            "moving.tif",
            "fixed.tif",
            "out.tif",
            transform_model="bspline",
            split_factor=1,
        )


def test_align_image_pair_bspline_rejects_multi_pass_solve_resolutions() -> None:
    with pytest.raises(ValueError, match="multi-pass"):
        alignment.align_image_pair(
            "moving.tif",
            "fixed.tif",
            "out.tif",
            transform_model="bspline",
            solve_resolutions=[6.0, 2.0],
        )


def test_align_image_pair_can_write_reference_grid_output(
    tmp_path,
    mocked_identity_registration,
) -> None:
    moving_transform = from_origin(0.0, 8.0, 1.0, 1.0)
    fixed_transform = from_origin(0.0, 8.0, 2.0, 2.0)
    moving_data = np.arange(64, dtype=np.int16).reshape(8, 8)
    fixed_data = np.arange(16, dtype=np.int16).reshape(4, 4)
    moving_path = write_test_raster(tmp_path / "source.tif", moving_data, transform=moving_transform)
    fixed_path = write_test_raster(tmp_path / "reference.tif", fixed_data, transform=fixed_transform)
    output_path = tmp_path / "aligned_reference_grid.tif"

    alignment.align_image_pair(
        moving_image_path=str(moving_path),
        fixed_image_path=str(fixed_path),
        output_image_path=str(output_path),
        temp_dir=str(tmp_path),
        output_on_moving_grid=False,
        use_edge_proxies=False,
    )

    with rasterio.open(output_path) as src:
        output = src.read(1)
        assert src.count == 1
        assert src.width == 4
        assert src.height == 4
        assert src.transform == fixed_transform
        assert src.crs.to_string() == DEFAULT_CRS
        assert src.nodata == -9999

    assert output.shape == (4, 4)
    assert np.any(output != -9999)
