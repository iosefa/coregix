# Vector Alignment Evaluation

Coregix can evaluate paired vector geometries against an alignment transform before an aligned raster is written. This is useful when vector features are available for quality control and you want to test registration parameters cheaply.

The workflow has two explicit steps:

1. Estimate the raster transform with `align-image-pair --dry-run`.
2. Evaluate paired fixed and moving vector features with `evaluate-vector-alignment`.

Dry-run mode is opt-in. Normal `align-image-pair` behavior still writes the aligned raster when `--dry-run` is not used.

## Estimate Only

Use `--dry-run` to estimate the transform and write only transform metadata:

```bash
align-image-pair \
  --moving-image /path/to/source.tif \
  --fixed-image /path/to/reference.tif \
  --output-transform-json /path/to/transform.coregix.json \
  --dry-run
```

In dry-run mode, `--output-transform-json` is required and `--output-image` is optional. Coregix does not write an aligned raster.

The transform JSON includes two CRS-coordinate matrices:

- `target_to_source`: maps target/output-reference coordinates to original moving/source coordinates sampled by Coregix.
- `source_to_target`: maps original moving/source coordinates into the fixed/aligned coordinate frame.

For vector QA, use `source_to_target`.

## Evaluate Paired Features

Run vector evaluation with a fixed vector file, a moving vector file, and the transform JSON:

```bash
evaluate-vector-alignment \
  --fixed-vector /path/to/fixed_features.gpkg \
  --moving-vector /path/to/moving_features.gpkg \
  --transform-json /path/to/transform.coregix.json \
  --id-field feature_id \
  --output-json /path/to/vector_rmse.json \
  --output-csv /path/to/vector_rmse.csv
```

The evaluator:

- loads paired geometries from the fixed and moving vector files
- pairs features by `--id-field`
- computes initial symmetric boundary RMSE before applying the transform
- applies `source_to_target` to the moving geometries
- computes aligned symmetric boundary RMSE after applying the transform
- reports improvement in raster CRS units and percent

`rmse_m` is retained as an alias for `aligned_rmse_m` for compatibility with earlier outputs.

The default rasterization pixel size is `0.5` CRS units. Change it with `--pixel-size` when your vectors should be evaluated at another scale:

```bash
evaluate-vector-alignment \
  --fixed-vector fixed_features.gpkg \
  --moving-vector moving_features.gpkg \
  --transform-json transform.coregix.json \
  --id-field feature_id \
  --pixel-size 1.0
```

Use `--padding` to control the local rasterization padding around each feature pair. The default is `8.0` CRS units.

## Python API

The same evaluation is available from Python:

```python
from coregix import evaluate_vector_alignment

result = evaluate_vector_alignment(
    fixed_vector_path="/path/to/fixed_features.gpkg",
    moving_vector_path="/path/to/moving_features.gpkg",
    transform_json_path="/path/to/transform.coregix.json",
    id_field="feature_id",
    output_json_path="/path/to/vector_rmse.json",
    output_csv_path="/path/to/vector_rmse.csv",
)

print(result.initial_rmse_m)
print(result.aligned_rmse_m)
print(result.improvement_m)
print(result.feature_count)
```

## Accept Then Write

If the RMSE is acceptable, apply the saved transform JSON without rerunning registration:

```bash
apply-coregix-transform \
  --moving-image /path/to/source.tif \
  --transform-json /path/to/transform.coregix.json \
  --output-image /path/to/aligned.tif
```

This keeps the workflow explicit: dry-run estimates a transform, vector evaluation checks it, and `apply-coregix-transform` performs only the raster resampling/write step.

## Requirements

Vector evaluation reads vector files through GDAL/OGR. Use a vector format supported by your GDAL installation, such as GeoPackage.
