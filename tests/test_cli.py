from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from coregix.cli import align_image_pair as align_cli
from coregix.cli import apply_coregix_transform as apply_cli
from coregix.cli import trim_edge_invalid as trim_cli
from coregix.cli import evaluate_vector_alignment as eval_cli


def test_console_script_name_has_no_legacy_vhr_prefix() -> None:
    pyproject = Path("pyproject.toml").read_text()
    legacy_command = "vhr" "-align-image-pair"

    assert 'align-image-pair = "coregix.cli.align_image_pair:main"' in pyproject
    assert 'apply-coregix-transform = "coregix.cli.apply_coregix_transform:main"' in pyproject
    assert 'evaluate-vector-alignment = "coregix.cli.evaluate_vector_alignment:main"' in pyproject
    assert legacy_command not in pyproject


def test_align_parser_defaults_use_current_registration_policy() -> None:
    args = align_cli.build_parser().parse_args(
        [
            "--moving-image",
            "source.tif",
            "--fixed-image",
            "reference.tif",
            "--output-image",
            "aligned.tif",
        ]
    )

    assert args.use_edge_proxies is True
    assert args.clip_fixed_to_moving is True
    assert args.output_on_moving_grid is True
    assert args.enforce_mutual_valid_mask is True
    assert args.split_factor == 0
    assert args.output_transform_json is None
    assert args.dry_run is False


def test_align_parser_boolean_flags_can_be_disabled() -> None:
    args = align_cli.build_parser().parse_args(
        [
            "--moving-image",
            "source.tif",
            "--fixed-image",
            "reference.tif",
            "--output-image",
            "aligned.tif",
            "--no-use-edge-proxies",
            "--no-clip-fixed-to-moving",
            "--no-output-on-moving-grid",
            "--no-enforce-mutual-valid-mask",
        ]
    )

    assert args.use_edge_proxies is False
    assert args.clip_fixed_to_moving is False
    assert args.output_on_moving_grid is False
    assert args.enforce_mutual_valid_mask is False


def test_align_parser_accepts_coarse_to_fine_solve_resolutions() -> None:
    args = align_cli.build_parser().parse_args(
        [
            "--moving-image",
            "source.tif",
            "--fixed-image",
            "reference.tif",
            "--output-image",
            "aligned.tif",
            "--solve-resolutions",
            "8,4,0",
        ]
    )

    assert args.solve_resolutions == [8.0, 4.0, None]


def test_align_cli_forwards_arguments_and_prints_json(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    moving = tmp_path / "source.tif"
    fixed = tmp_path / "reference.tif"
    output = tmp_path / "aligned.tif"
    transform_json = tmp_path / "aligned.coregix.json"
    moving.touch()
    fixed.touch()
    captured_kwargs = {}

    def fake_align_image_pair(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            output_image_path=str(output),
            temp_dir=None,
            output_transform_json_path=kwargs.get("output_transform_json_path"),
            dry_run=kwargs.get("dry_run", False),
        )

    monkeypatch.setattr(align_cli, "align_image_pair", fake_align_image_pair)

    exit_code = align_cli.main(
        [
            "--moving-image",
            str(moving),
            "--fixed-image",
            str(fixed),
            "--output-image",
            str(output),
            "--output-transform-json",
            str(transform_json),
            "--moving-band-index",
            "1",
            "--fixed-band-index",
            "2",
            "--split-factor",
            "2",
            "--solve-resolutions",
            "8,4,0",
            "--trim-edge-invalid",
            "--edge-trim-depth",
            "3",
            "--edge-trim-invalid-below",
            "-3000",
            "--no-output-on-moving-grid",
            "--no-use-edge-proxies",
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["moving_image_path"] == str(moving)
    assert captured_kwargs["fixed_image_path"] == str(fixed)
    assert captured_kwargs["output_image_path"] == str(output)
    assert captured_kwargs["output_transform_json_path"] == str(transform_json)
    assert captured_kwargs["moving_band_index"] == 1
    assert captured_kwargs["fixed_band_index"] == 2
    assert captured_kwargs["split_factor"] == 2
    assert captured_kwargs["solve_resolution"] is None
    assert captured_kwargs["solve_resolutions"] == [8.0, 4.0, None]
    assert captured_kwargs["trim_edge_invalid"] is True
    assert captured_kwargs["edge_trim_depth"] == 3
    assert captured_kwargs["edge_trim_invalid_below"] == -3000
    assert captured_kwargs["output_on_moving_grid"] is False
    assert captured_kwargs["use_edge_proxies"] is False
    assert captured_kwargs["dry_run"] is False

    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "output_image_path": str(output),
        "temp_dir": None,
        "output_transform_json_path": str(transform_json),
        "dry_run": False,
    }


def test_align_cli_dry_run_does_not_require_output_image(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    moving = tmp_path / "source.tif"
    fixed = tmp_path / "reference.tif"
    transform_json = tmp_path / "transform.json"
    moving.touch()
    fixed.touch()
    captured_kwargs = {}

    def fake_align_image_pair(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            output_image_path=None,
            temp_dir=None,
            output_transform_json_path=kwargs.get("output_transform_json_path"),
            dry_run=kwargs.get("dry_run", False),
        )

    monkeypatch.setattr(align_cli, "align_image_pair", fake_align_image_pair)

    exit_code = align_cli.main(
        [
            "--moving-image",
            str(moving),
            "--fixed-image",
            str(fixed),
            "--output-transform-json",
            str(transform_json),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["output_image_path"] is None
    assert captured_kwargs["output_transform_json_path"] == str(transform_json)
    assert captured_kwargs["dry_run"] is True
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "output_image_path": None,
        "temp_dir": None,
        "output_transform_json_path": str(transform_json),
        "dry_run": True,
    }


def test_align_cli_dry_run_requires_transform_json(tmp_path) -> None:
    moving = tmp_path / "source.tif"
    fixed = tmp_path / "reference.tif"
    moving.touch()
    fixed.touch()

    with pytest.raises(SystemExit):
        align_cli.main(
            [
                "--moving-image",
                str(moving),
                "--fixed-image",
                str(fixed),
                "--dry-run",
            ]
        )


def test_align_cli_normal_mode_requires_output_image(tmp_path) -> None:
    moving = tmp_path / "source.tif"
    fixed = tmp_path / "reference.tif"
    moving.touch()
    fixed.touch()

    with pytest.raises(SystemExit):
        align_cli.main(
            [
                "--moving-image",
                str(moving),
                "--fixed-image",
                str(fixed),
            ]
        )


def test_align_cli_rejects_invalid_split_factor(tmp_path) -> None:
    moving = tmp_path / "source.tif"
    fixed = tmp_path / "reference.tif"
    moving.touch()
    fixed.touch()

    with pytest.raises(SystemExit):
        align_cli.main(
            [
                "--moving-image",
                str(moving),
                "--fixed-image",
                str(fixed),
                "--output-image",
                str(tmp_path / "aligned.tif"),
                "--split-factor",
                "-1",
            ]
        )


def test_align_cli_parser_accepts_transform_model(tmp_path, monkeypatch, capsys) -> None:
    moving = tmp_path / "source.tif"
    fixed = tmp_path / "reference.tif"
    output = tmp_path / "aligned.tif"
    moving.touch()
    fixed.touch()
    captured_kwargs = {}

    def fake_align_image_pair(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            output_image_path=kwargs["output_image_path"],
            temp_dir=None,
            output_transform_json_path=None,
            transform_metadata=None,
            dry_run=False,
        )

    monkeypatch.setattr(align_cli, "align_image_pair", fake_align_image_pair)

    assert align_cli.main(
        [
            "--moving-image",
            str(moving),
            "--fixed-image",
            str(fixed),
            "--output-image",
            str(output),
            "--transform-model",
            "bspline",
        ]
    ) == 0
    assert captured_kwargs["transform_model"] == "bspline"


def test_align_cli_bspline_rejects_transform_json(tmp_path) -> None:
    moving = tmp_path / "source.tif"
    fixed = tmp_path / "reference.tif"
    moving.touch()
    fixed.touch()

    with pytest.raises(SystemExit):
        align_cli.main(
            [
                "--moving-image",
                str(moving),
                "--fixed-image",
                str(fixed),
                "--output-image",
                str(tmp_path / "aligned.tif"),
                "--transform-model",
                "bspline",
                "--output-transform-json",
                str(tmp_path / "transform.json"),
            ]
        )


def test_align_cli_bspline_rejects_split_factor(tmp_path) -> None:
    moving = tmp_path / "source.tif"
    fixed = tmp_path / "reference.tif"
    moving.touch()
    fixed.touch()

    with pytest.raises(SystemExit):
        align_cli.main(
            [
                "--moving-image",
                str(moving),
                "--fixed-image",
                str(fixed),
                "--output-image",
                str(tmp_path / "aligned.tif"),
                "--transform-model",
                "bspline",
                "--split-factor",
                "1",
            ]
        )


def test_align_cli_bspline_rejects_dry_run(tmp_path) -> None:
    moving = tmp_path / "source.tif"
    fixed = tmp_path / "reference.tif"
    moving.touch()
    fixed.touch()

    with pytest.raises(SystemExit):
        align_cli.main(
            [
                "--moving-image",
                str(moving),
                "--fixed-image",
                str(fixed),
                "--transform-model",
                "bspline",
                "--dry-run",
                "--output-transform-json",
                str(tmp_path / "transform.json"),
            ]
        )


def test_align_cli_bspline_rejects_multi_pass_solve_resolutions(tmp_path) -> None:
    moving = tmp_path / "source.tif"
    fixed = tmp_path / "reference.tif"
    moving.touch()
    fixed.touch()

    with pytest.raises(SystemExit):
        align_cli.main(
            [
                "--moving-image",
                str(moving),
                "--fixed-image",
                str(fixed),
                "--output-image",
                str(tmp_path / "aligned.tif"),
                "--transform-model",
                "bspline",
                "--solve-resolutions",
                "4,2",
            ]
        )


def test_apply_coregix_transform_parser() -> None:
    args = apply_cli.build_parser().parse_args(
        [
            "--moving-image",
            "source.tif",
            "--transform-json",
            "transform.json",
            "--output-image",
            "aligned.tif",
            "--allow-moving-image-mismatch",
            "--trim-edge-invalid",
            "--edge-trim-depth",
            "3",
            "--edge-trim-invalid-below",
            "-1000",
        ]
    )

    assert args.moving_image == "source.tif"
    assert args.transform_json == "transform.json"
    assert args.output_image == "aligned.tif"
    assert args.allow_moving_image_mismatch is True
    assert args.trim_edge_invalid is True
    assert args.edge_trim_depth == 3
    assert args.edge_trim_invalid_below == -1000


def test_apply_coregix_transform_cli_forwards_arguments(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    moving = tmp_path / "source.tif"
    transform = tmp_path / "transform.json"
    output = tmp_path / "aligned.tif"
    moving.touch()
    transform.touch()
    captured_kwargs = {}

    def fake_apply_coregix_transform(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            output_image_path=kwargs["output_image_path"],
            transform_json_path=kwargs["transform_json_path"],
            output_width=10,
            output_height=20,
            output_transform=[1.0, 0.0, 100.0, 0.0, -1.0, 200.0],
        )

    monkeypatch.setattr(apply_cli, "apply_coregix_transform", fake_apply_coregix_transform)

    exit_code = apply_cli.main(
        [
            "--moving-image",
            str(moving),
            "--transform-json",
            str(transform),
            "--output-image",
            str(output),
            "--allow-moving-image-mismatch",
            "--trim-edge-invalid",
            "--edge-trim-depth",
            "3",
            "--edge-trim-invalid-below",
            "-1000",
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["moving_image_path"] == str(moving)
    assert captured_kwargs["transform_json_path"] == str(transform)
    assert captured_kwargs["output_image_path"] == str(output)
    assert captured_kwargs["allow_moving_image_mismatch"] is True
    assert captured_kwargs["trim_edge_invalid"] is True
    assert captured_kwargs["edge_trim_depth"] == 3
    assert captured_kwargs["edge_trim_invalid_below"] == -1000
    summary = json.loads(capsys.readouterr().out)
    assert summary["output_image_path"] == str(output)
    assert summary["transform_json_path"] == str(transform)
    assert summary["output_width"] == 10
    assert summary["output_height"] == 20


def test_evaluate_vector_alignment_parser() -> None:
    args = eval_cli.build_parser().parse_args(
        [
            "--fixed-vector",
            "fixed.gpkg",
            "--moving-vector",
            "moving.gpkg",
            "--transform-json",
            "transform.json",
            "--id-field",
            "feature_id",
        ]
    )

    assert args.fixed_vector == "fixed.gpkg"
    assert args.moving_vector == "moving.gpkg"
    assert args.transform_json == "transform.json"
    assert args.id_field == "feature_id"
    assert args.pixel_size == 0.5
    assert args.padding == 8.0


def test_evaluate_vector_alignment_cli_prints_initial_and_aligned_rmse(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixed = tmp_path / "fixed.gpkg"
    moving = tmp_path / "moving.gpkg"
    transform = tmp_path / "transform.json"
    fixed.touch()
    moving.touch()
    transform.touch()

    def fake_evaluate_vector_alignment(**kwargs):
        return SimpleNamespace(
            rmse_m=1.5,
            initial_rmse_m=4.0,
            aligned_rmse_m=1.5,
            improvement_m=2.5,
            improvement_percent=62.5,
            feature_count=3,
            output_json_path=kwargs.get("output_json_path"),
            output_csv_path=kwargs.get("output_csv_path"),
        )

    monkeypatch.setattr(
        eval_cli, "evaluate_vector_alignment", fake_evaluate_vector_alignment
    )

    exit_code = eval_cli.main(
        [
            "--fixed-vector",
            str(fixed),
            "--moving-vector",
            str(moving),
            "--transform-json",
            str(transform),
            "--id-field",
            "feature_id",
            "--output-json",
            str(tmp_path / "rmse.json"),
            "--output-csv",
            str(tmp_path / "rmse.csv"),
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "rmse_m": 1.5,
        "initial_rmse_m": 4.0,
        "aligned_rmse_m": 1.5,
        "improvement_m": 2.5,
        "improvement_percent": 62.5,
        "feature_count": 3,
        "output_json_path": str(tmp_path / "rmse.json"),
        "output_csv_path": str(tmp_path / "rmse.csv"),
    }


def test_trim_cli_requires_output_or_in_place(tmp_path) -> None:
    input_image = tmp_path / "aligned.tif"
    input_image.touch()

    with pytest.raises(SystemExit):
        trim_cli.main(["--input-image", str(input_image)])
