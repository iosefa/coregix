# Large Rasters

## Chunked Alignment

Use `--split-factor` for large moving rasters that are expensive to transform in one pass. The value controls the number of chunks as `2 ** split_factor`.

Quadrant-style execution:

```bash
vhr-align-image-pair \
  --moving-image /path/to/moving_large.tif \
  --fixed-image /path/to/fixed.tif \
  --output-image /path/to/aligned_large.tif \
  --split-factor 2
```

Python:

```python
from coregix import align_image_pair

result = align_image_pair(
    moving_image_path="/path/to/moving_large.tif",
    fixed_image_path="/path/to/fixed.tif",
    output_image_path="/path/to/aligned_large.tif",
    split_factor=2,
)
```

## Choosing a Split Factor

Start with `split_factor=0`. Increase the value when memory use or runtime becomes impractical.

| Use case | Suggested value |
| --- | --- |
| Small to medium rasters | `0` |
| Large rasters needing moderate chunking | `1` or `2` |
| Very large rasters | `3` or higher |

Higher split factors create more chunks and more overhead. Use the smallest value that keeps processing stable.

## Coarser Registration Solve

`--solve-resolution` runs the registration solve on a coarser grid while still writing the final output at the requested output grid:

```bash
vhr-align-image-pair \
  --moving-image moving_large.tif \
  --fixed-image fixed.tif \
  --output-image aligned_large.tif \
  --split-factor 2 \
  --solve-resolution 2.0
```

The value is in raster CRS units. For a projected CRS in meters, `2.0` means a 2-meter solve grid.

## Notes

- `split_factor` affects chunked solve and transform application.
- Band metadata and descriptions are copied from the moving raster when possible.
- Output nodata defaults to moving nodata, then fixed nodata, then `0`.
