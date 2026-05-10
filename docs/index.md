# Coregix

[![PyPI](https://img.shields.io/pypi/v/coregix.svg)](https://pypi.org/project/coregix/)
[![PyPI Downloads](https://static.pepy.tech/badge/coregix)](https://pepy.tech/projects/coregix)
[![Docker Pulls](https://img.shields.io/docker/pulls/iosefa/coregix?logo=docker&label=pulls)](https://hub.docker.com/r/iosefa/coregix)
[![Docs](https://img.shields.io/github/actions/workflow/status/iosefa/coregix/docs.yml?branch=main&label=docs)](https://github.com/iosefa/coregix/actions/workflows/docs.yml)
[![Contributors](https://img.shields.io/github/contributors/iosefa/coregix.svg?label=contributors)](https://github.com/iosefa/coregix/graphs/contributors)
[![License](https://img.shields.io/github/license/iosefa/coregix)](https://github.com/iosefa/coregix/blob/main/LICENSE)

**Pairwise raster coregistration for geospatial imagery.**

Coregix is a Python package and command-line tool for coregistering one geospatial raster to another. It estimates a geometric transform from a source raster to a reference raster, applies that transform to the source image, and writes a coregistered GeoTIFF while preserving geospatial metadata and multi-band outputs.

Coregix is intended for residual image-to-image alignment: cases where rasters are already georeferenced and close to alignment, but still visibly offset at the pixel level. It does not replace orthorectification, sensor model correction, or CRS reprojection.

## Registration Model

The default registration is area-based. Coregix estimates a translation followed by a rigid transform using mutual-information optimization, then applies the result to all source bands.

Outputs can be written on either the source-raster grid or the reference-raster grid. Keeping the source grid is useful when the source image belongs to an existing stack; writing to the reference grid is useful when preparing matched rasters for comparison.

## Input Expectations

Coregix expects:

- source and reference rasters in the same CRS
- enough overlapping valid pixels for registration
- useful raster masks or nodata values for invalid regions
- imagery that is close enough for translation/rigid registration to converge

Large rasters can be processed in chunks. Edge cleanup is available for outputs with invalid border artifacts after resampling.

## First Alignment

Use `--moving-image` for the source raster and `--fixed-image` for the reference raster:

```bash
vhr-align-image-pair \
  --moving-image /path/to/source.tif \
  --fixed-image /path/to/reference.tif \
  --output-image /path/to/aligned.tif
```

The same alignment can be run from Python:

```python
from coregix import align_image_pair

result = align_image_pair(
    moving_image_path="/path/to/source.tif",
    fixed_image_path="/path/to/reference.tif",
    output_image_path="/path/to/aligned.tif",
)

print(result.output_image_path)
```

## Next Steps

- [Installation](installation.md): set up the package and documentation environment.
- [Concepts](concepts.md): read about grids, masks, edge proxies, and chunking.
- [Align Image Pairs](usage/alignment.md): run the CLI or Python API.
- [Large Rasters](usage/large-rasters.md): use chunked transform application.
- [Edge Trimming](usage/edge-trimming.md): clean invalid border artifacts.
- [API Reference](api/index.md): inspect generated API documentation.
