from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from coregix.cli import align_image_pair as align_cli
from coregix.cli import trim_edge_invalid as trim_cli


def test_console_script_name_has_no_legacy_vhr_prefix() -> None:
    pyproject = Path("pyproject.toml").read_text()
    legacy_command = "vhr" "-align-image-pair"

    assert 'align-image-pair = "coregix.cli.align_image_pair:main"' in pyproject
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
    moving.touch()
    fixed.touch()
    captured_kwargs = {}

    def fake_align_image_pair(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(output_image_path=str(output), temp_dir=None)

    monkeypatch.setattr(align_cli, "align_image_pair", fake_align_image_pair)

    exit_code = align_cli.main(
        [
            "--moving-image",
            str(moving),
            "--fixed-image",
            str(fixed),
            "--output-image",
            str(output),
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

    summary = json.loads(capsys.readouterr().out)
    assert summary == {"output_image_path": str(output), "temp_dir": None}


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


def test_trim_cli_requires_output_or_in_place(tmp_path) -> None:
    input_image = tmp_path / "aligned.tif"
    input_image.touch()

    with pytest.raises(SystemExit):
        trim_cli.main(["--input-image", str(input_image)])
