# Coregix

[![PyPI](https://img.shields.io/pypi/v/coregix.svg)](https://pypi.org/project/coregix/)
[![PyPI Downloads](https://static.pepy.tech/badge/coregix)](https://pepy.tech/projects/coregix)
[![Docker Pulls](https://img.shields.io/docker/pulls/iosefa/coregix?logo=docker&label=pulls)](https://hub.docker.com/r/iosefa/coregix)
[![Tests](https://img.shields.io/github/actions/workflow/status/iosefa/coregix/tests.yml?branch=main&label=tests)](https://github.com/iosefa/coregix/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/iosefa/coregix/docs.yml?branch=main&label=docs)](https://github.com/iosefa/coregix/actions/workflows/docs.yml)
[![Contributors](https://img.shields.io/github/contributors/iosefa/coregix.svg?label=contributors)](https://github.com/iosefa/coregix/graphs/contributors)
[![License](https://img.shields.io/github/license/iosefa/coregix)](https://github.com/iosefa/coregix/blob/main/LICENSE)

Coregix provides pairwise raster coregistration for geospatial imagery.

Coregix coregisters a source raster to a reference raster while preserving geospatial metadata and multi-band outputs. By default, it estimates a translation followed by a rigid transform using mutual-information optimization, then applies the resulting transform to produce a coregistered GeoTIFF.

Current scope:
- pairwise GeoTIFF coregistration CLI and Python API
- edge-proxy registration for cross-sensor structural alignment
- chunked transform application for large source rasters
- experimental single-pass B-spline nonrigid raster alignment
- optional postprocess trimming of invalid edge artifacts

## Install

### Conda environment

```bash
conda env create -f environment.yml
conda activate coregix
```

This installs the runtime stack and the package in editable mode.

### Editable install into an existing environment

```bash
pip install -e .
```

The installed CLI entrypoint is:

```bash
align-image-pair --help
```

You can also run the module directly:

```bash
python -m coregix.cli.align_image_pair --help
```

### Docker

Build the image from the repository root:

```bash
docker build -t coregix .
```

Release images are published to Docker Hub as `iosefa/coregix`.

Run the CLI with a mounted data directory:

```bash
docker run --rm \
  -v "$PWD:/data" \
  iosefa/coregix:latest \
  --moving-image /data/source.tif \
  --fixed-image /data/reference.tif \
  --output-image /data/aligned.tif
```

If you built the image locally, use `coregix` instead of `iosefa/coregix:latest`.

## CLI usage

### Coregister a source image to a reference image

```bash
align-image-pair \
  --moving-image /path/to/source.tif \
  --fixed-image /path/to/reference.tif \
  --output-image /path/to/aligned.tif
```

By default this:
- registers on edge-proxy images
- writes the result on the source-raster grid
- uses no chunking (`--split-factor 0`)

### Experimental nonrigid alignment

Coregix can run an experimental B-spline nonrigid registration for cases where a
single global shift/rotation leaves local residual distortion:

```bash
align-image-pair \
  --moving-image /path/to/source.tif \
  --fixed-image /path/to/reference.tif \
  --output-image /path/to/aligned_bspline.tif \
  --transform-model bspline
```

This mode currently supports only direct single-pass raster output. It cannot be
used with `--dry-run`, `--output-transform-json`, `apply-coregix-transform`,
`evaluate-vector-alignment`, `--split-factor > 0`, or multi-pass
`--solve-resolutions`. A single `--solve-resolution` or one-value
`--solve-resolutions` is allowed. Because a B-spline transform can locally warp
image geometry, inspect the output visually and compare it with the default
rigid result before using it in downstream analysis.

### Use chunking for large source rasters

`--split-factor` controls chunked transform application as `2^k` total chunks:
- `0`: no split
- `1`: halves
- `2`: quadrants
- `3`: octants

Example with quadrants:

```bash
align-image-pair \
  --moving-image /path/to/source_large.tif \
  --fixed-image /path/to/reference.tif \
  --output-image /path/to/aligned_large.tif \
  --split-factor 2
```

### Use coarse-to-fine registration for large initial offsets

`--solve-resolutions` runs multiple registration solves from coarse to fine,
then writes the final raster once from the original source image. Use `0` for
the reference-raster/native solve resolution.

`--solve-resolutions` works with `--split-factor 0` for whole-image multi-pass solves when memory permits, or with higher split factors for chunked solves.

`--solve-resolution` is deprecated and remains available for single-pass
compatibility. Prefer `--solve-resolutions`, even for one solve.

```bash
align-image-pair \
  --moving-image /path/to/source_large.tif \
  --fixed-image /path/to/reference.tif \
  --output-image /path/to/aligned_large.tif \
  --split-factor 2 \
  --solve-resolutions 8,4,0.5
```

### Write transform metadata for vector QA

Use `--output-transform-json` to write an optional sidecar with the final coordinate transform:

```bash
align-image-pair \
  --moving-image /path/to/source.tif \
  --fixed-image /path/to/reference.tif \
  --output-image /path/to/aligned.tif \
  --output-transform-json /path/to/aligned.coregix.json
```

The JSON includes `target_to_source` and `source_to_target` affine matrices in raster CRS coordinates. Use `source_to_target` to transform original moving vector geometries into the fixed/aligned coordinate frame.

Use `--dry-run` to estimate and write only the transform metadata without writing the aligned raster:

```bash
align-image-pair \
  --moving-image /path/to/source.tif \
  --fixed-image /path/to/reference.tif \
  --output-transform-json /path/to/transform.coregix.json \
  --dry-run
```

Then evaluate paired vector features with:

```bash
evaluate-vector-alignment \
  --fixed-vector /path/to/fixed_features.gpkg \
  --moving-vector /path/to/moving_features.gpkg \
  --transform-json /path/to/transform.coregix.json \
  --id-field feature_id \
  --output-json /path/to/vector_rmse.json \
  --output-csv /path/to/vector_rmse.csv
```

If the transform passes QA, apply the saved transform without rerunning registration:

```bash
apply-coregix-transform \
  --moving-image /path/to/source.tif \
  --transform-json /path/to/transform.coregix.json \
  --output-image /path/to/aligned.tif
```

### Remove invalid edge artifacts after alignment

`--trim-edge-invalid` runs a raster-space cleanup pass after alignment and sets edge artifacts to nodata.

Example:

```bash
align-image-pair \
  --moving-image /path/to/source_large.tif \
  --fixed-image /path/to/reference.tif \
  --output-image /path/to/aligned_large_edgefixed.tif \
  --split-factor 2 \
  --trim-edge-invalid \
  --edge-trim-depth 8 \
  --edge-trim-invalid-below -3000
```

The edge-trim thresholds are dataset-specific. `--edge-trim-invalid-below` is useful when interpolation artifacts are not equal to the dataset nodata value.

## Python usage

### Basic alignment

```python
from coregix import align_image_pair

result = align_image_pair(
    moving_image_path="/path/to/source.tif",
    fixed_image_path="/path/to/reference.tif",
    output_image_path="/path/to/aligned.tif",
)

print(result.output_image_path)
```

### Experimental nonrigid alignment

```python
from coregix import align_image_pair

result = align_image_pair(
    moving_image_path="/path/to/source.tif",
    fixed_image_path="/path/to/reference.tif",
    output_image_path="/path/to/aligned_bspline.tif",
    transform_model="bspline",
    solve_resolutions=[2.0],
)

print(result.output_image_path)
```

B-spline alignment is raster-output-only. It does not write Coregix transform
JSON and is not compatible with saved-transform application, vector RMSE
evaluation, chunking, dry-run mode, or multi-pass solve sequences.

### Large raster with chunking and edge cleanup

```python
from coregix import align_image_pair

result = align_image_pair(
    moving_image_path="/path/to/source_large.tif",
    fixed_image_path="/path/to/reference.tif",
    output_image_path="/path/to/aligned_large_edgefixed.tif",
    split_factor=2,
    trim_edge_invalid=True,
    edge_trim_depth=8,
    edge_trim_invalid_below=-3000,
)

print(result.output_image_path)
```

### Dry-run transform metadata and vector evaluation

```python
from coregix import align_image_pair, apply_coregix_transform, evaluate_vector_alignment

transform = align_image_pair(
    moving_image_path="/path/to/source.tif",
    fixed_image_path="/path/to/reference.tif",
    output_transform_json_path="/path/to/transform.coregix.json",
    dry_run=True,
    solve_resolutions=[6.0, 2.0],
)

print(transform.output_image_path)  # None in dry-run mode
print(transform.output_transform_json_path)

qa = evaluate_vector_alignment(
    fixed_vector_path="/path/to/fixed_features.gpkg",
    moving_vector_path="/path/to/moving_features.gpkg",
    transform_json_path="/path/to/transform.coregix.json",
    id_field="feature_id",
)

print(qa.initial_rmse_m)  # RMSE before applying the transform
print(qa.aligned_rmse_m)  # RMSE after applying the transform
print(qa.improvement_m)

if qa.aligned_rmse_m < 1.0:
    applied = apply_coregix_transform(
        moving_image_path="/path/to/source.tif",
        transform_json_path="/path/to/transform.coregix.json",
        output_image_path="/path/to/aligned.tif",
    )
    print(applied.output_image_path)
```

## Notes

- `--dry-run` is explicit opt-in; normal alignment writes `--output-image`.
- Vector alignment evaluation reads vector files through GDAL/OGR, so your environment must include GDAL Python bindings for formats such as GeoPackage.
- `split_factor` changes only transform application, not the registration model for the default rigid workflow. It is not supported with `--transform-model bspline`.
- `split_factor=2` is the direct replacement for the previous quadrant-based large-raster path.
- If needed, you can select separate registration bands with `moving_band_index` and `fixed_band_index` in Python or `--moving-band-index` and `--fixed-band-index` in the CLI.
