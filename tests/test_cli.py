from __future__ import annotations

import json
from pathlib import Path

import pytest

from coregix.cli import align_image_pair as align_cli
from coregix.cli import trim_edge_invalid as trim_cli


def test_console_script_name_has_no_legacy_vhr_prefix():
    pyproject = Path("pyproject.toml").read_text()
    assert 'align-image-pair = "coregix.cli.align_image_pair:main"' in pyproject
    assert "vhr-align-image-pair" not in pyproject


def test_align_cli_forwards_joint_options(monkeypatch, capsys):
    calls = []

    def align(*args, **kwargs):
        calls.append((args, kwargs))
        return ["aligned.tif"]

    monkeypatch.setattr(align_cli, "align_image_pair", align)
    assert align_cli.main([
        "--input-images", "source.tif", "--output-images", "aligned.tif",
        "--fixed-images", "reference.tif", "--options", '{"local_model": "none"}',
    ]) == 0
    assert calls == [
        ((["source.tif"], ["aligned.tif"]),
         {"fixed_images": ["reference.tif"], "local_model": "none"}),
    ]
    assert json.loads(capsys.readouterr().out) == {"output_images": ["aligned.tif"]}


def test_align_cli_rejects_nonobject_options():
    with pytest.raises(SystemExit):
        align_cli.main(["--input-images", "in.tif", "--output-images", "out.tif",
                        "--options", "[]"])


def test_trim_cli_requires_output_or_in_place(tmp_path):
    source = tmp_path / "aligned.tif"
    source.touch()
    with pytest.raises(SystemExit):
        trim_cli.main(["--input-image", str(source)])
