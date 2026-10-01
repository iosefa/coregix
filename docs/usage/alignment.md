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
  "temp_dir": null
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

## Continue when overlap is insufficient

Set `warn_and_skip_non_overlap_images=True` in Python or pass
`--warn-and-skip-non-overlap-images` on the CLI to continue when the rasters have
no spatial overlap or too few usable pixels for `min_valid_fraction`. Coregix
emits a `RuntimeWarning` and copies the unchanged source to the requested output
path. This retains its grid, pixels, masks, band metadata, and overviews; output
grid/nodata overrides and edge trimming do not apply to this fallback.

The default is `False`, which raises an error. With chunking, individual chunks
without coverage remain excluded from the solve; the fallback applies when all
chunks lack usable coverage. Registration backend failures and invalid inputs
remain errors. A copied output has **not** been coregistered, even when its name
ends in `_aligned.tif`.

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
