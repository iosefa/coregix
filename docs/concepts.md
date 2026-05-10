# Concepts

## Fixed and Moving Images

Coregix follows the common image-registration naming convention:

- **Fixed image**: the reference raster that defines the target alignment.
- **Moving image**: the raster that is transformed to match the fixed image.

The default output is written on the moving-image grid. This preserves the moving raster's dimensions, transform, band count, and most band metadata while updating the pixel values in the overlap region.

Use `--no-output-on-moving-grid` or `output_on_moving_grid=False` when you need the output on the fixed-image grid instead.

## Edge-proxy Registration

By default, Coregix registers edge-proxy images instead of raw pixel intensities. The edge proxy is computed from local image gradients over valid pixels. This is useful when two rasters share structural boundaries but have different spectral or radiometric characteristics.

Disable this behavior with:

```bash
vhr-align-image-pair ... --no-use-edge-proxies
```

or in Python:

```python
align_image_pair(..., use_edge_proxies=False)
```

## Valid Masks and Nodata

Coregix builds registration masks from raster masks and nodata values. You can override nodata values when source metadata is missing or incorrect:

```bash
vhr-align-image-pair \
  --moving-image moving.tif \
  --fixed-image fixed.tif \
  --output-image aligned.tif \
  --moving-nodata 0 \
  --fixed-nodata 0
```

`--min-valid-fraction` controls the minimum valid-data fraction required for registration.

## Chunking

`split_factor` controls chunked solve and transform application for large rasters. The total number of chunks is `2 ** split_factor`:

| `split_factor` | Chunks |
| --- | --- |
| `0` | no split |
| `1` | 2 chunks |
| `2` | 4 chunks |
| `3` | 8 chunks |

Chunking reduces memory pressure during transform application. It does not change the public alignment call.

## Solve Resolution

`solve_resolution` lets you run the registration solve on a coarser grid, expressed in raster CRS units. This can reduce registration cost for large, high-resolution rasters:

```bash
vhr-align-image-pair \
  --moving-image moving.tif \
  --fixed-image fixed.tif \
  --output-image aligned.tif \
  --solve-resolution 2.0
```
