# Coregix

Coregix provides elastix-based pairwise raster alignment for geospatial imagery.

Current scope:
- pairwise raster alignment CLI and Python API
- edge-proxy registration for structural cross-sensor alignment
- chunked transform application for large moving rasters
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

## CLI usage

### Align a moving image to a fixed image

```bash
vhr-align-image-pair \
  --moving-image /path/to/moving.tif \
  --fixed-image /path/to/fixed.tif \
  --output-image /path/to/aligned.tif
```

By default this:
- registers on edge-proxy images
- writes the result on the moving-image grid
- uses no chunking (`--split-factor 0`)

### Use chunking for large moving rasters

`--split-factor` controls chunked transform application as `2^k` total chunks:
- `0`: no split
- `1`: halves
- `2`: quadrants
- `3`: octants

Example with quadrants:

```bash
vhr-align-image-pair \
  --moving-image /path/to/moving_large.tif \
  --fixed-image /path/to/fixed.tif \
  --output-image /path/to/aligned_large.tif \
  --split-factor 2
```

### Remove invalid edge artifacts after alignment

`--trim-edge-invalid` runs a raster-space cleanup pass after alignment and sets edge artifacts to nodata.

Example:

```bash
vhr-align-image-pair \
  --moving-image /path/to/moving_large.tif \
  --fixed-image /path/to/fixed.tif \
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
    moving_image_path="/path/to/moving.tif",
    fixed_image_path="/path/to/fixed.tif",
    output_image_path="/path/to/aligned.tif",
)

print(result.output_image_path)
```

### Large raster with chunking and edge cleanup

```python
from coregix import align_image_pair

result = align_image_pair(
    moving_image_path="/path/to/moving_large.tif",
    fixed_image_path="/path/to/fixed.tif",
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
