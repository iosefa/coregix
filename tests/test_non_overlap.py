from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from coregix.pipelines import alignment, alignment_large_main
from tests.helpers import write_test_raster


def raster_pair(tmp_path, kind):
    data = np.arange(256, dtype=np.int16).reshape(16, 16) + 1
    moving = np.stack([data, data + 1000])
    fixed = data.copy()
    transform = from_origin(100, 200, 1, 1)
    if kind == "disjoint":
        transform = from_origin(1000, 2000, 1, 1)
    elif kind == "reference_nodata":
        fixed[:] = -9999
    elif kind == "source_nodata":
        moving[:] = -9999
    elif kind == "mutual_nodata":
        fixed[:, :8] = -9999
        moving[:, :, 8:] = -9999
    elif kind == "sparse":
        fixed[:] = -9999
        fixed[8, 8] = 5  # Nonzero support, but below the required 5%.
    source = write_test_raster(tmp_path / "source.tif", moving)
    reference = write_test_raster(tmp_path / "reference.tif", fixed, transform=transform)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False), rasterio.open(source, "r+") as dst:
        mask = np.full((16, 16), 255, dtype=np.uint8)
        mask[0] = 0
        dst.write_mask(mask)  # External .msk must survive the renamed output.
        dst.update_tags(scene="P001")
        dst.set_band_description(1, "blue")
        dst.update_tags(1, wavelength="450")
        dst.scales = (0.01, 0.02)
        dst.offsets = (1, 2)
        dst.build_overviews([2], rasterio.enums.Resampling.nearest)
    return source, reference


@pytest.mark.parametrize("split_factor", [0, 1])
@pytest.mark.parametrize("clip", [False, True])
@pytest.mark.parametrize("kind", ["disjoint", "reference_nodata", "source_nodata", "mutual_nodata", "sparse"])
def test_skip_preserves_source_at_requested_output(tmp_path, monkeypatch, split_factor, clip, kind):
    source, reference = raster_pair(tmp_path, kind)
    output = tmp_path / "outputs/scene_aligned.tif"
    calls = []

    def unexpected_solve(**kwargs):
        calls.append(kwargs)
        raise AssertionError("Insufficient coverage must not enter the registration backend")

    for module in (alignment, alignment_large_main):
        monkeypatch.setattr(module, "estimate_elastix_transform", unexpected_solve)
    with pytest.warns(RuntimeWarning, match="Skipping coregistration") as recorded:
        result = alignment.align_image_pair(
            str(source), str(reference), str(output), split_factor=split_factor,
            warn_and_skip_non_overlap_images=True, min_valid_fraction=0.05,
            clip_fixed_to_moving=clip, output_on_moving_grid=False,
            trim_edge_invalid=True, output_nodata=-1000, temp_dir=str(tmp_path),
        )
    assert not calls
    assert len(recorded) == 1
    assert str(source) in str(recorded[0].message) and str(reference) in str(recorded[0].message)
    assert result.output_image_path == str(output) and result.temp_dir is None
    assert not list(tmp_path.glob("vhr_align_*"))
    with rasterio.open(source) as src, rasterio.open(output) as dst:
        np.testing.assert_array_equal(dst.read(), src.read())
        np.testing.assert_array_equal(dst.read_masks(), src.read_masks())
        assert (dst.crs, dst.transform, dst.nodata) == (src.crs, src.transform, src.nodata)
        assert dst.tags() == src.tags()
        assert dst.tags(1) == src.tags(1)
        assert dst.descriptions == src.descriptions
        assert dst.scales == src.scales and dst.offsets == src.offsets
        assert src.overviews(1) == [2]
        assert dst.overviews(1) == []
    # The workflow must be able to add overviews without overriding COG protection.
    with rasterio.open(output, "r+") as dst:
        dst.build_overviews([2], rasterio.enums.Resampling.nearest)
    with rasterio.open(output) as dst:
        assert dst.overviews(1) == [2]


@pytest.mark.parametrize("split_factor", [0, 1])
@pytest.mark.parametrize("kind", ["disjoint", "reference_nodata", "source_nodata", "mutual_nodata", "sparse"])
def test_default_still_raises_for_insufficient_overlap(tmp_path, split_factor, kind):
    source, reference = raster_pair(tmp_path, kind)
    output = tmp_path / "aligned.tif"
    with pytest.raises(ValueError, match="overlap|support"):
        alignment.align_image_pair(str(source), str(reference), str(output),
                                   split_factor=split_factor, min_valid_fraction=0.05)
    assert not output.exists()


@pytest.mark.parametrize("split_factor", [0, 1])
def test_backend_failure_is_not_treated_as_non_overlap(tmp_path, monkeypatch, split_factor):
    source, reference = raster_pair(tmp_path, "valid")

    def broken_backend(**kwargs):
        raise RuntimeError("registration backend failed")

    for module in (alignment, alignment_large_main):
        monkeypatch.setattr(module, "estimate_elastix_transform", broken_backend)
    output = tmp_path / "aligned.tif"
    with pytest.raises((RuntimeError, ValueError), match="Elastix registration failed|No chunk produced"):
        alignment.align_image_pair(str(source), str(reference), str(output),
                                   split_factor=split_factor, warn_and_skip_non_overlap_images=True)
    assert not output.exists()


@pytest.mark.parametrize("split_factor", [0, 1])
def test_invalid_band_is_not_treated_as_non_overlap(tmp_path, split_factor):
    source, reference = raster_pair(tmp_path, "disjoint")
    output = tmp_path / "aligned.tif"
    with pytest.raises(ValueError, match="Requested source band"):
        alignment.align_image_pair(str(source), str(reference), str(output), moving_band_index=10,
                                   split_factor=split_factor, warn_and_skip_non_overlap_images=True)
    assert not output.exists()


@pytest.mark.parametrize("split_factor", [0, 1])
def test_skip_can_retain_temp_directory(tmp_path, split_factor):
    source, reference = raster_pair(tmp_path, "reference_nodata")
    with pytest.warns(RuntimeWarning, match="Skipping coregistration"):
        result = alignment.align_image_pair(
            str(source), str(reference), str(tmp_path / "aligned.tif"),
            split_factor=split_factor, warn_and_skip_non_overlap_images=True,
            temp_dir=str(tmp_path), keep_temp_dir=True,
        )
    assert Path(result.temp_dir).is_dir()
