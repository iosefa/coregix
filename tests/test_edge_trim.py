from __future__ import annotations

import numpy as np
import pytest
import rasterio

from coregix.postprocess.edge_trim import (
    _dilate_mask_square,
    _invalid_mask,
    trim_edge_invalid_pixels,
)
from tests.helpers import write_test_raster


def test_invalid_mask_combines_nodata_and_thresholds() -> None:
    data = np.array([[0.0, -9999.0], [-12.0, 50.0]], dtype=np.float32)

    mask = _invalid_mask(
        data,
        nodata_value=-9999.0,
        invalid_below=-10.0,
        invalid_above=40.0,
    )

    np.testing.assert_array_equal(mask, [[False, True], [True, True]])


def test_dilate_mask_square_expands_by_radius() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True

    dilated = _dilate_mask_square(mask, radius=1)

    assert dilated.sum() == 9
    assert np.all(dilated[1:4, 1:4])
    assert not dilated[0, 0]


def test_trim_edge_invalid_pixels_sets_dilated_invalid_region_to_nodata(tmp_path) -> None:
    data = np.stack(
        [
            np.full((5, 5), 10, dtype=np.int16),
            np.full((5, 5), 20, dtype=np.int16),
        ]
    )
    data[0, 2, 2] = -9999
    input_path = write_test_raster(tmp_path / "input.tif", data, nodata=-9999)
    output_path = tmp_path / "trimmed.tif"

    result = trim_edge_invalid_pixels(
        str(input_path),
        output_image_path=str(output_path),
        edge_depth=1,
        row_chunk_size=2,
        col_chunk_size=2,
    )

    assert result.output_image_path == str(output_path)
    assert result.nodata_value == -9999.0
    assert result.pixels_trimmed == 8

    with rasterio.open(output_path) as src:
        trimmed = src.read()
        assert src.count == 2
        assert src.nodata == -9999

    assert np.all(trimmed[:, 1:4, 1:4] == -9999)
    assert trimmed[0, 0, 0] == 10
    assert trimmed[1, 0, 0] == 20


def test_trim_edge_invalid_pixels_requires_invalid_criteria(tmp_path) -> None:
    data = np.full((4, 4), 1, dtype=np.int16)
    input_path = write_test_raster(tmp_path / "input.tif", data, nodata=None)

    with pytest.raises(ValueError, match="nodata_value"):
        trim_edge_invalid_pixels(str(input_path), output_image_path=str(tmp_path / "out.tif"))
