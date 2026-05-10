# Large Rasters

Large rasters can be expensive to transform in a single pass. Coregix supports chunked processing so the registration solve and final output writing can be split into smaller pieces.

Chunking is controlled with `--split-factor`. The total number of chunks is `2 ** split_factor`.

## Chunked Alignment

Start without chunking. Add `--split-factor` when memory use or runtime becomes impractical:

```bash
vhr-align-image-pair \
  --moving-image /path/to/source_large.tif \
  --fixed-image /path/to/reference.tif \
  --output-image /path/to/aligned_large.tif \
  --split-factor 2
```

Python:

```python
from coregix import align_image_pair

result = align_image_pair(
    moving_image_path="/path/to/source_large.tif",
    fixed_image_path="/path/to/reference.tif",
    output_image_path="/path/to/aligned_large.tif",
    split_factor=2,
    clip_fixed_to_moving=True,
    enforce_mutual_valid_mask=True,
)
```

## Choosing a Split Factor

Use the smallest split factor that keeps processing stable.

| `split_factor` | Chunks | Typical use |
| --- | --- | --- |
| `0` | 1 | small and medium rasters |
| `1` | 2 | moderate memory reduction |
| `2` | 4 | quadrant-style processing |
| `3` | 8 | very large rasters |

Higher values create more chunks and more overhead. They can also fail if individual chunks do not contain enough valid, informative overlap for registration.

## Coarser Registration Solve

`--solve-resolution` runs the registration solve on a coarser grid while still writing the final output on the requested output grid. The value is expressed in raster CRS units.

For a projected CRS in meters, `--solve-resolution 2.0` uses an approximate 2-meter solve grid:

```bash
vhr-align-image-pair \
  --moving-image source_large.tif \
  --fixed-image reference.tif \
  --output-image aligned_large.tif \
  --split-factor 2 \
  --solve-resolution 2.0
```

This can reduce registration cost for high-resolution rasters. Use a solve resolution that preserves the spatial structure needed for alignment.

## Coarse-to-fine Alignment

Area-based registration works best when the source and reference rasters are already close. For difficult pairs, a coarse-to-fine approach can be more reliable than a single full-resolution run.

One practical pattern is:

1. create lower-resolution source and reference products
2. coregister the lower-resolution source to the lower-resolution reference
3. use the coarse result to bring the full-resolution source closer to the reference
4. run Coregix again at full resolution to refine the alignment

The first pass handles the larger residual offset on a simpler image pair. The second pass starts from a closer alignment, giving the area-based registration a better chance of converging on the fine-scale correction.

## Grid and Metadata Behavior

Chunked output follows the same grid rules as standard alignment:

- source-grid output is the default
- reference-grid output is selected with `--no-output-on-moving-grid`
- band metadata and descriptions are copied from the source raster when possible
- output nodata defaults to source nodata, then reference nodata, then `0`

Chunking changes how the transform is estimated and applied internally; it does not change the output file format or the public API.
