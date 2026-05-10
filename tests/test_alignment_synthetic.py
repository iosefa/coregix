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

    def fake_apply_elastix_transform_array(
        moving_image,
        transform_parameter_object,
        *,
        log_to_console=False,
    ):
        return np.asarray(moving_image, dtype=np.float32)

    monkeypatch.setattr(alignment, "estimate_elastix_transform", fake_estimate_elastix_transform)
    monkeypatch.setattr(alignment, "deformation_field_from_transform", fake_deformation_field_from_transform)
    monkeypatch.setattr(alignment, "apply_elastix_transform_array", fake_apply_elastix_transform_array)
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
