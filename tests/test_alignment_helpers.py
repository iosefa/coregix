from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

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
