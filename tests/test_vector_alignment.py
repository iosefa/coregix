from __future__ import annotations

import csv
import json

import pytest

from coregix.evaluation import evaluate_vector_alignment


def _write_polygon_gpkg(path, *, layer_name: str, feature_id: str, x_offset: float) -> None:
    ogr = pytest.importorskip("osgeo.ogr")
    osr = pytest.importorskip("osgeo.osr")

    driver = ogr.GetDriverByName("GPKG")
    if path.exists():
        path.unlink()
    ds = driver.CreateDataSource(str(path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32610)
    layer = ds.CreateLayer(layer_name, srs, ogr.wkbPolygon)
    field = ogr.FieldDefn("feature_id", ogr.OFTString)
    layer.CreateField(field)

    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in (
        (x_offset, 0.0),
        (x_offset + 4.0, 0.0),
        (x_offset + 4.0, 4.0),
        (x_offset, 4.0),
        (x_offset, 0.0),
    ):
        ring.AddPoint(x, y)
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)

    feat = ogr.Feature(layer.GetLayerDefn())
    feat.SetField("feature_id", feature_id)
    feat.SetGeometry(polygon)
    layer.CreateFeature(feat)
    ds = None


def test_evaluate_vector_alignment_reports_initial_and_aligned_rmse(tmp_path) -> None:
    fixed = tmp_path / "fixed.gpkg"
    moving = tmp_path / "moving.gpkg"
    transform_json = tmp_path / "transform.json"
    output_json = tmp_path / "rmse.json"
    output_csv = tmp_path / "rmse.csv"

    _write_polygon_gpkg(fixed, layer_name="fixed_features", feature_id="a", x_offset=0.0)
    _write_polygon_gpkg(moving, layer_name="moving_features", feature_id="a", x_offset=2.0)
    transform_json.write_text(
        json.dumps(
            {
                "source_to_target": {
                    "matrix": [
                        [1.0, 0.0, -2.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_vector_alignment(
        fixed_vector_path=str(fixed),
        moving_vector_path=str(moving),
        transform_json_path=str(transform_json),
        id_field="feature_id",
        pixel_size=1.0,
        padding=2.0,
        output_json_path=str(output_json),
        output_csv_path=str(output_csv),
    )

    assert result.feature_count == 1
    assert result.rmse_m == result.aligned_rmse_m
    assert result.initial_rmse_m > 0.0
    assert result.aligned_rmse_m == pytest.approx(0.0)
    assert result.improvement_m == pytest.approx(result.initial_rmse_m)
    assert result.improvement_percent == pytest.approx(100.0)
    assert result.per_feature[0].initial_rmse_m == pytest.approx(result.initial_rmse_m)
    assert result.per_feature[0].aligned_rmse_m == pytest.approx(0.0)

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["initial_rmse_m"] == pytest.approx(result.initial_rmse_m)
    assert payload["aligned_rmse_m"] == pytest.approx(result.aligned_rmse_m)
    assert payload["improvement_percent"] == pytest.approx(100.0)
    assert payload["per_feature"][0]["initial_rmse_m"] == pytest.approx(result.initial_rmse_m)

    with open(output_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["feature_id"] == "a"
    assert float(rows[0]["aligned_rmse_m"]) == pytest.approx(0.0)
    assert rows[0]["improvement_percent"] == "100.000000"
