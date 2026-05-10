# Align Image Pairs

## CLI

The main command coregisters a source raster to a reference raster:

```bash
vhr-align-image-pair \
  --moving-image /path/to/source.tif \
  --fixed-image /path/to/reference.tif \
  --output-image /path/to/aligned.tif
```

By default, this command:

- registers on edge-proxy images
- clips the reference domain to source-raster bounds
- uses the mutual valid-data overlap for elastix masks
- writes the coregistered output on the source-raster grid
- applies no chunking (`--split-factor 0`)

The command prints a JSON summary:

```json
{
  "output_image_path": "/path/to/aligned.tif",
  "temp_dir": null
}
```

## Python API

```python
from coregix import align_image_pair

result = align_image_pair(
    moving_image_path="/path/to/source.tif",
    fixed_image_path="/path/to/reference.tif",
    output_image_path="/path/to/aligned.tif",
)

print(result.output_image_path)
```

## Band Selection

Use `band_index` when the same 0-based band should be used from both rasters:

```bash
vhr-align-image-pair \
  --moving-image moving.tif \
  --fixed-image fixed.tif \
  --output-image aligned.tif \
  --band-index 2
```

Use separate source and reference band indexes when the best registration signal is in different bands:

```bash
vhr-align-image-pair \
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

## Output Grid

The default output grid is the source-raster grid:

```bash
vhr-align-image-pair \
  --moving-image source.tif \
  --fixed-image reference.tif \
  --output-image aligned.tif \
  --output-on-moving-grid
```

To write on the reference-raster grid:

```bash
vhr-align-image-pair \
  --moving-image source.tif \
  --fixed-image reference.tif \
  --output-image aligned_on_reference_grid.tif \
  --no-output-on-moving-grid
```

## Temporary Files

Coregix creates temporary working files during registration. Keep them for debugging with:

```bash
vhr-align-image-pair \
  --moving-image source.tif \
  --fixed-image reference.tif \
  --output-image aligned.tif \
  --keep-temp-dir
```

Use `--temp-dir` to choose the parent directory for those files.
