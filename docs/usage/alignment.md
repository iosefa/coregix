# Align Image Pairs

This page covers the standard pairwise coregistration case: one source raster, one reference raster, and one coregistered GeoTIFF output.

The CLI keeps the registration library's argument names:

- `--moving-image` is the source raster that will be transformed.
- `--fixed-image` is the reference raster used for alignment.

## Basic CLI Alignment

```bash
align-image-pair \
  --moving-image /path/to/source.tif \
  --fixed-image /path/to/reference.tif \
  --output-image /path/to/aligned.tif
```

By default, the CLI:

- estimates registration on edge-proxy images
- clips the reference domain to the source-raster bounds
- uses the mutual valid-data overlap for registration masks
- writes the result on the source-raster grid
- uses no chunking (`--split-factor 0`)

The command prints a JSON summary:

```json
{
  "output_image_path": "/path/to/aligned.tif",
  "temp_dir": null,
  "output_transform_json_path": null,
  "dry_run": false
}
```

## Basic Python Alignment

The same settings can be used from Python:

```python
from coregix import align_image_pair

result = align_image_pair(
    moving_image_path="/path/to/source.tif",
    fixed_image_path="/path/to/reference.tif",
    output_image_path="/path/to/aligned.tif",
)

print(result.output_image_path)
```

The CLI and Python API use the same registration defaults.

## Experimental Nonrigid Alignment

Use `--transform-model bspline` to try a B-spline nonrigid correction when a
global rigid transform improves one area but leaves local residual distortion
elsewhere:

```bash
align-image-pair \
  --moving-image source.tif \
  --fixed-image reference.tif \
  --output-image aligned_bspline.tif \
  --transform-model bspline
```

Python:

```python
align_image_pair(
    moving_image_path="source.tif",
    fixed_image_path="reference.tif",
    output_image_path="aligned_bspline.tif",
    transform_model="bspline",
)
```

This mode is experimental and raster-output-only. It cannot be used with
`--dry-run`, `--output-transform-json`, `apply-coregix-transform`,
`evaluate-vector-alignment`, `--split-factor > 0`, or multi-pass
`--solve-resolutions`. A single `--solve-resolution` or one-value
`--solve-resolutions` is allowed.

Use this mode when the default rigid result leaves local residual distortion
that cannot be fixed with one global translation/rotation. Do not use it as a
default replacement for rigid alignment: the deformation can improve one part of
an image while introducing shape changes elsewhere. Compare the B-spline output
against the rigid output and inspect local control features before treating it as
analysis-ready.

When a vector RMSE report is required, run the default rigid workflow with
`--output-transform-json` and `evaluate-vector-alignment`. B-spline transforms
are currently represented only by the raster deformation field used during the
alignment run, so Coregix does not export a reusable matrix JSON for vector QA or
later transform application.

## Registration Bands

Registration is estimated from one band in each raster. Use `--band-index` when the same 0-based band should be used from both rasters:

```bash
align-image-pair \
  --moving-image source.tif \
  --fixed-image reference.tif \
  --output-image aligned.tif \
  --band-index 2
```

Use separate band indexes when the best registration signal is in different bands:

```bash
align-image-pair \
  --moving-image source.tif \
  --fixed-image reference.tif \
  --output-image aligned.tif \
  --moving-band-index 0 \
  --fixed-band-index 3
```

The same options are available in Python:

```python
align_image_pair(
    moving_image_path="source.tif",
    fixed_image_path="reference.tif",
    output_image_path="aligned.tif",
    moving_band_index=0,
    fixed_band_index=3,
)
```

All source bands are transformed after the registration model is estimated.

## Output Grid

The default output grid is the source-raster grid:

```bash
align-image-pair \
  --moving-image source.tif \
  --fixed-image reference.tif \
  --output-image aligned.tif \
  --output-on-moving-grid
```

Use this when the aligned raster needs to remain compatible with a source image stack.

Write on the reference-raster grid with:

```bash
align-image-pair \
  --moving-image source.tif \
  --fixed-image reference.tif \
  --output-image aligned_on_reference_grid.tif \
  --no-output-on-moving-grid
```

Use this when the output should match the reference raster's extent, transform, and pixel grid.

## Transform JSON

Coregix can optionally write a JSON sidecar with the final coordinate transform:

```bash
align-image-pair \
  --moving-image source.tif \
  --fixed-image reference.tif \
  --output-image aligned.tif \
  --output-transform-json aligned.coregix.json
```

This is disabled by default. The JSON includes `target_to_source` and `source_to_target` homogeneous affine matrices in raster CRS coordinates. `target_to_source` describes how Coregix samples the original moving raster for each output/reference coordinate. Use `source_to_target` to transform original moving vector geometries into the fixed/aligned coordinate frame for RMSE checks. Transform JSON is supported for the default rigid workflow only, not for `--transform-model bspline`.

The same option is available in Python with `output_transform_json_path`.

Use dry-run mode when you only want to estimate and save the transform, without writing the aligned raster:

```bash
align-image-pair \
  --moving-image source.tif \
  --fixed-image reference.tif \
  --output-transform-json transform.coregix.json \
  --dry-run
```

In normal mode, `--output-image` is required and Coregix writes the aligned raster. In dry-run mode, `--output-transform-json` is required and `--output-image` is optional.

Evaluate paired vector features against a transform JSON with:

```bash
evaluate-vector-alignment \
  --fixed-vector fixed_features.gpkg \
  --moving-vector moving_features.gpkg \
  --transform-json transform.coregix.json \
  --id-field feature_id \
  --output-json vector_rmse.json \
  --output-csv vector_rmse.csv
```

The evaluator pairs features by `--id-field` and reports both the initial symmetric boundary RMSE and the aligned RMSE after applying `source_to_target` to the moving features. `rmse_m` is retained as an alias for the aligned/post-transform RMSE. See [Vector Alignment Evaluation](vector-alignment.md) for full CLI and Python usage.

If the transform passes QA, apply the saved transform without rerunning registration:

```bash
apply-coregix-transform \
  --moving-image source.tif \
  --transform-json transform.coregix.json \
  --output-image aligned.tif
```

## Nodata Overrides

Coregix reads raster masks and nodata metadata when building registration masks. If the source metadata is missing or incorrect, provide nodata values explicitly:

```bash
align-image-pair \
  --moving-image source.tif \
  --fixed-image reference.tif \
  --output-image aligned.tif \
  --moving-nodata 0 \
  --fixed-nodata 0 \
  --output-nodata 0
```

`--output-nodata` controls the value written into invalid output pixels.

## Debugging Registration

Temporary registration images and masks are normally deleted after the command finishes. Keep them when diagnosing a failed or unexpected registration:

```bash
align-image-pair \
  --moving-image source.tif \
  --fixed-image reference.tif \
  --output-image aligned.tif \
  --keep-temp-dir
```

Use `--temp-dir` to choose the parent directory for temporary files, and `--log-to-console` to print registration backend logs.
