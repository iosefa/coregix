# Coregix

[![PyPI](https://img.shields.io/pypi/v/coregix.svg)](https://pypi.org/project/coregix/)
[![PyPI Downloads](https://static.pepy.tech/badge/coregix)](https://pepy.tech/projects/coregix)
[![Docker Pulls](https://img.shields.io/docker/pulls/iosefa/coregix?logo=docker&label=pulls)](https://hub.docker.com/r/iosefa/coregix)
[![Docs](https://img.shields.io/github/actions/workflow/status/iosefa/coregix/docs.yml?branch=main&label=docs)](https://github.com/iosefa/coregix/actions/workflows/docs.yml)
[![Contributors](https://img.shields.io/github/contributors/iosefa/coregix.svg?label=contributors)](https://github.com/iosefa/coregix/graphs/contributors)
[![License](https://img.shields.io/github/license/iosefa/coregix)](https://github.com/iosefa/coregix/blob/main/LICENSE)

Coregix provides pairwise raster coregistration for geospatial imagery.

Coregix coregisters a source raster to a reference raster while preserving geospatial metadata and multi-band outputs. By default, it estimates a translation followed by a rigid transform using mutual-information optimization, then applies the resulting transform to produce a coregistered GeoTIFF.

Current scope:
- pairwise GeoTIFF coregistration CLI and Python API
- edge-proxy registration for cross-sensor structural alignment
- chunked transform application for large source rasters
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
vhr-align-image-pair --help
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
vhr-align-image-pair \
  --moving-image /path/to/source.tif \
  --fixed-image /path/to/reference.tif \
  --output-image /path/to/aligned.tif
```

By default this:
- registers on edge-proxy images
- writes the result on the source-raster grid
- uses no chunking (`--split-factor 0`)

### Use chunking for large source rasters

`--split-factor` controls chunked transform application as `2^k` total chunks:
- `0`: no split
- `1`: halves
- `2`: quadrants
- `3`: octants

Example with quadrants:

```bash
vhr-align-image-pair \
  --moving-image /path/to/source_large.tif \
  --fixed-image /path/to/reference.tif \
  --output-image /path/to/aligned_large.tif \
  --split-factor 2
```

### Remove invalid edge artifacts after alignment

`--trim-edge-invalid` runs a raster-space cleanup pass after alignment and sets edge artifacts to nodata.

Example:

```bash
vhr-align-image-pair \
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

## Notes

- `split_factor` changes only transform application, not the registration model.
- `split_factor=2` is the direct replacement for the previous quadrant-based large-raster path.
- If needed, you can select separate registration bands with `moving_band_index` and `fixed_band_index` in Python or `--moving-band-index` and `--fixed-band-index` in the CLI.
