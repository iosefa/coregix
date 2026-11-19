# Coregix

[![PyPI](https://img.shields.io/pypi/v/coregix.svg)](https://pypi.org/project/coregix/)
[![PyPI Downloads](https://static.pepy.tech/badge/coregix)](https://pepy.tech/projects/coregix)
[![Docker Pulls](https://img.shields.io/docker/pulls/iosefa/coregix?logo=docker&label=pulls)](https://hub.docker.com/r/iosefa/coregix)
[![Tests](https://img.shields.io/github/actions/workflow/status/iosefa/coregix/tests.yml?branch=main&label=tests)](https://github.com/iosefa/coregix/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/iosefa/coregix/docs.yml?branch=main&label=docs)](https://github.com/iosefa/coregix/actions/workflows/docs.yml)
[![Contributors](https://img.shields.io/github/contributors/iosefa/coregix.svg?label=contributors)](https://github.com/iosefa/coregix/graphs/contributors)
[![License](https://img.shields.io/github/license/iosefa/coregix)](https://github.com/iosefa/coregix/blob/main/LICENSE)

Coregix aligns overlapping geospatial rasters through a Python API and command-line interface. Align one image to a fixed reference, or solve a network of overlapping images jointly using image-wide transforms and local deformation meshes.

Tie points connect corresponding locations across images. Coregix first solves global corrections, then fits a local displacement mesh to the remaining offsets. The combined corrections are applied to every output band.

- ORB feature matching and Elastix mutual-information matching
- Global translation, similarity, and affine models
- Bilinear and piecewise affine local deformation models
- Fixed references, per-image movement weights, and robust tie-point losses
- Reusable tie-point caches, GIS tie-point exports, and Cloud-Optimized GeoTIFF output

## Install

```bash
conda env create -f environment.yml
conda activate coregix
```

This installs the runtime dependencies and Coregix in editable mode. Python 3.10 or later is required.

For an existing environment, install GDAL and matching Python bindings, then install from the repository root:

```bash
pip install -e .
```

## Python usage

Align an image to a fixed reference:

```python
from coregix import align_image_pair

outputs = align_image_pair(
    input_images=["source.tif"],
    output_images=["aligned.tif"],
    fixed_images=["reference.tif"],
)

print(outputs)  # ["aligned.tif"]
```

`input_images` contains the moving rasters; `output_images` specifies their destinations. Fixed references stay unchanged and have no output entries. The return value contains output paths in input order.

To align multiple images jointly without a fixed reference:

```python
outputs = align_image_pair(
    ["tile_a.tif", "tile_b.tif", "tile_c.tif"],
    "aligned/",
    global_model="translation",
    local_model="piecewise_affine",
)
```

The output folder receives one file per moving raster. The Python API also accepts input folders, glob patterns, and output filename templates.

### Common options

| Option | Purpose |
| --- | --- |
| `global_model` | Choose `none`, `translation`, `similarity`, or `affine`. |
| `local_model` | Choose `none`, `bilinear`, or `piecewise_affine`. |
| `input_images_bands`, `fixed_images_bands` | Select one-based bands for matching. All bands are retained in outputs. |
| `tie_save_path`, `tie_load_path` | Save and reuse selected tie points. |
| `tie_save_crs_path` | Export tie points to a GIS vector file. |
| `resolution`, `tap` | Set output pixel spacing and align bounds to the target pixel grid. |
| `window_size`, `save_as_cog` | Configure output tiles and Cloud-Optimized GeoTIFF writing. |

Moving-image pairs use ORB by default; pairs involving fixed references use Elastix. Set `local_model="none"` when only an image-wide correction is needed.

## Command-line usage

```bash
align-image-pair \
  --input-images source.tif \
  --output-images aligned.tif \
  --fixed-images reference.tif \
  --options '{"global_model":"translation","local_model":"piecewise_affine"}'
```

`--options` accepts algorithm keyword arguments as a JSON object. The command prints `{"output_images": ["aligned.tif"]}`.

```bash
align-image-pair --help
```

### Docker

Build and run the CLI with a mounted data directory:

```bash
docker build -t coregix .
docker run --rm -v "$PWD:/data" coregix \
  --input-images /data/source.tif \
  --output-images /data/aligned.tif \
  --fixed-images /data/reference.tif
```

## Input requirements

- Rasters must share a coordinate reference system and have unique basenames.
- Alignment requires overlapping valid pixels with enough shared image structure to find matches.
- Raster masks and nodata metadata define valid matching regions.
- Matching band numbers are one-based; multiple selected bands are averaged for matching.

## Documentation

See [alignment usage](docs/usage/alignment.md), [large-raster processing](docs/usage/large-rasters.md), and the [API reference](docs/api/alignment.md) for details. Standalone [edge trimming](docs/usage/edge-trimming.md) is available to clean invalid border artifacts after alignment.
