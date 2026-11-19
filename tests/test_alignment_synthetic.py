from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from coregix import align_image_pair
from tests.helpers import write_test_raster

joint = importlib.import_module("coregix._joint.joint_coregistration.joint_coregistration")


@pytest.fixture
def raster_pair(tmp_path):
    data = np.stack([
        np.arange(1024, dtype=np.int16).reshape(32, 32),
        np.full((32, 32), 2000, dtype=np.int16),
    ])
    fixed = write_test_raster(tmp_path / "fixed.tif", data,
                              transform=from_origin(0, 32, 1, 1))
    moving = write_test_raster(tmp_path / "moving.tif", data,
                               transform=from_origin(2, 32, 1, 1))
    cache = tmp_path / "ties.json"
    cache.write_text(json.dumps({"tie_points": [{
        "image_1": "fixed", "image_2": "moving",
        "points": [[[8, 8], [8, 8]], [[24, 8], [24, 8]],
                   [[8, 24], [8, 24]], [[24, 24], [24, 24]]],
    }]}))
    return str(fixed), str(moving), str(cache), data


@pytest.mark.parametrize("global_model", ["translation", "similarity", "affine"])
def test_fixed_reference_anchors_global_alignment(tmp_path, raster_pair, global_model):
    fixed, moving, cache, data = raster_pair
    output = str(tmp_path / "aligned.tif")
    before = open(fixed, "rb").read()
    paths = align_image_pair(
        [moving], [output], fixed_images=[fixed], tie_load_path=cache,
        global_model=global_model, local_model="none", tie_sampling_method="grid",
    )
    assert paths == [output]
    assert open(fixed, "rb").read() == before
    with rasterio.open(output) as result, rasterio.open(fixed) as reference:
        np.testing.assert_allclose(tuple(result.transform), tuple(reference.transform), atol=1e-6)
        np.testing.assert_array_equal(result.read(), data)
        assert result.count == 2
        assert result.nodata == -9999
        assert result.crs == reference.crs


def test_unanchored_network_splits_translation(tmp_path, raster_pair):
    fixed, moving, cache, _ = raster_pair
    outputs = [str(tmp_path / "out_fixed.tif"), str(tmp_path / "out_moving.tif")]
    align_image_pair([fixed, moving], outputs, tie_load_path=cache,
                     local_model="none", tie_sampling_method="grid")
    for output in outputs:
        with rasterio.open(output) as result:
            assert result.transform.c == pytest.approx(1, abs=1e-6)


@pytest.mark.parametrize("model", ["bilinear", "piecewise_affine"])
def test_local_model_writes_readable_multiband_output(tmp_path, raster_pair, model):
    fixed, moving, cache, _ = raster_pair
    output = str(tmp_path / "local.tif")
    align_image_pair(
        [moving], [output], fixed_images=[fixed], tie_load_path=cache,
        global_model="none", local_model=model, local_grid_spacing=16,
        tie_sampling_method="grid", local_smoothness_weight=0.1,
        local_bending_weight=0.1,
    )
    with rasterio.open(output) as result:
        assert result.count == 2
        assert result.width >= 32
        assert result.height >= 32
        assert result.read_masks(1).any()
        assert np.isfinite(result.read()).all()


def test_overlap_spread_reuses_selected_ties(tmp_path, raster_pair, monkeypatch):
    fixed, moving, cache, _ = raster_pair
    monkeypatch.setattr(joint, "_extract_grid_pixel_matches",
                        lambda *args, **kwargs: pytest.fail("Cached ties must bypass detection"))
    output = str(tmp_path / "spread.tif")
    align_image_pair([fixed, moving], [str(tmp_path / "other.tif"), output],
                     tie_load_path=cache, local_model="none",
                     tie_max_samples=4, tie_min_samples=1)
    with rasterio.open(output) as result:
        assert result.transform.c == pytest.approx(1, abs=1e-6)


def test_zero_strength_preserves_native_transform(tmp_path, raster_pair):
    fixed, moving, _, data = raster_pair
    output = str(tmp_path / "unchanged.tif")
    align_image_pair([moving], [output], fixed_images=[fixed],
                     global_tie_alignment_strength=0, local_tie_alignment_strength=0)
    with rasterio.open(output) as result, rasterio.open(moving) as source:
        assert result.transform == source.transform
        np.testing.assert_array_equal(result.read(), data)


def test_overviews_and_output_resolution(tmp_path, raster_pair):
    fixed, moving, cache, _ = raster_pair
    output = str(tmp_path / "overview.tif")
    align_image_pair([moving], [output], fixed_images=[fixed], tie_load_path=cache,
                     local_model="none", tie_sampling_method="grid", resolution=2,
                     tap=True, build_overviews=True, window_scales=(2, 4))
    with rasterio.open(output) as result:
        assert result.res == (2, 2)
        assert result.overviews(1) == [2, 4]


def test_reference_cannot_be_overwritten(raster_pair):
    fixed, moving, cache, _ = raster_pair
    with pytest.raises(ValueError, match="in-place"):
        align_image_pair([moving], [fixed], fixed_images=[fixed], tie_load_path=cache)


def test_three_images_are_solved_jointly(tmp_path, raster_pair):
    fixed, moving, cache, data = raster_pair
    third = write_test_raster(tmp_path / "third.tif", data,
                              transform=from_origin(4, 32, 1, 1))
    cache = Path(cache)
    model = json.loads(cache.read_text())
    model["tie_points"].append({
        "image_1": "moving", "image_2": "third",
        "points": [[[8, 8], [8, 8]]],
    })
    # Cache every overlap, including a consistent direct first-to-third tie.
    model["tie_points"].append({
        "image_1": "fixed", "image_2": "third",
        "points": [[[8, 8], [8, 8]]],
    })
    cache.write_text(json.dumps(model))
    outputs = [str(tmp_path / f"out_{i}.tif") for i in range(3)]
    assert align_image_pair([str(fixed), str(moving), str(third)], outputs,
                            tie_load_path=str(cache), tie_sampling_method="grid",
                            local_model="none", tie_robust_loss="none") == outputs
    for output in outputs:
        with rasterio.open(output) as result:
            assert result.transform.c == pytest.approx(2, abs=1e-6)


@pytest.mark.parametrize("model", ["bilinear", "piecewise_affine"])
def test_local_mesh_reduces_residual_and_holds_reference_fixed(raster_pair, model):
    fixed, moving, _, _ = raster_pair
    joint = importlib.import_module("coregix._joint.joint_coregistration.joint_coregistration")
    infos = joint._use_common_coordinate_frame({
        "fixed": joint._read_image_info("fixed", str(fixed)),
        "moving": joint._read_image_info("moving", str(moving)),
    })
    points = np.array([[8., 8., 8., 8.], [24., 24., 24., 24.]])
    meshes = joint._solve_local_alignment(
        infos, {("fixed", "moving"): points}, "none",
        {name: np.empty(0) for name in infos}, model,
        dict.fromkeys(infos, 1.), 1., 16., .01, .01, 500., "none", 1., False,
        {"fixed"},
    )
    np.testing.assert_array_equal(meshes["fixed"].displacement, 0)
    reference_xy = joint._pixels_to_map(infos["fixed"].transform, points[:, :2])
    moving_xy = joint._pixels_to_map(infos["moving"].transform, points[:, 2:])
    corrected = moving_xy + joint._evaluate_mesh(meshes["moving"], moving_xy, model)
    assert np.linalg.norm(corrected - reference_xy) < .05 * np.linalg.norm(moving_xy - reference_xy)


@pytest.mark.parametrize("method", ["orb", "elastix"])
def test_real_window_matcher_recovers_known_offset(method):
    joint = importlib.import_module("coregix._joint.joint_coregistration.joint_coregistration")
    rng = np.random.default_rng(13)
    valid = np.ones((192, 192), dtype=bool)
    if method == "orb":
        import cv2
        fixed = rng.integers(0, 256, valid.shape, dtype=np.uint8)
        moving = np.roll(fixed, 3, axis=1)
        points = joint._match_orb_windows(
            fixed, moving, valid, cv2.ORB_create(nfeatures=1000),
            cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True),
        )
        assert len(points) > 20
    else:
        pytest.importorskip("itk")
        from scipy.ndimage import gaussian_filter
        from coregix._joint.joint_coregistration.elastix import match_window
        fixed = gaussian_filter(rng.normal(size=valid.shape), 3).astype(np.float32)
        moving = np.roll(fixed, 3, axis=1)
        points = match_window(fixed, moving, valid)
        assert len(points) == 1
    np.testing.assert_allclose(np.median(points[:, 2:] - points[:, :2], axis=0),
                               [3, 0], atol=.5)


@pytest.mark.parametrize("options", [
    {"input_images_bands": [0]}, {"fixed_images_bands": []},
    {"local_grid_spacing": 0}, {"tie_search_radius": "128"},
    {"global_model": "rigid"}, {"tie_max_samples": 0},
])
def test_invalid_options_are_rejected(tmp_path, raster_pair, options):
    fixed, moving, _, _ = raster_pair
    with pytest.raises(ValueError):
        align_image_pair([moving], [str(tmp_path / "invalid.tif")],
                         fixed_images=[fixed], **options)
