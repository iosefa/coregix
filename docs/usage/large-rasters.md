# Large Rasters

Large rasters can be expensive to transform in a single pass. Coregix supports chunked processing so the registration solve and final output writing can be split into smaller pieces.

Chunking is controlled with `--split-factor`. The total number of chunks is `2 ** split_factor`.

## Chunked Alignment

Start without chunking. Add `--split-factor` when memory use or runtime becomes impractical:

```bash
align-image-pair \
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

## Coarse-to-fine Registration Solve

`--solve-resolutions` runs one or more registration solves from coarse to fine while still writing the final output on the requested output grid. Values are expressed in raster CRS units. Use `0` for the native/reference solve resolution.

For a projected CRS in meters, `--solve-resolutions 8,4,0.5` first solves on an approximate 8-meter grid, refines on a 4-meter grid, then refines again on a 0.5-meter grid:

```bash
align-image-pair \
  --moving-image source_large.tif \
  --fixed-image reference.tif \
  --output-image aligned_large.tif \
  --split-factor 2 \
  --solve-resolutions 8,4,0.5
```

Each pass refines the previous transform, and the final raster is sampled once from the original source image. This can make large-offset alignments more stable without stacking multiple resampling steps.

`--solve-resolution` is deprecated and remains available only for single-pass compatibility. Prefer `--solve-resolutions`, even for one solve, for example `--solve-resolutions 2.0`.

## Manual Coarse-to-fine Alignment

The in-memory `--solve-resolutions` workflow should be the default coarse-to-fine approach. A manual two-stage pattern can still be useful when you want to inspect or keep the intermediate coarse result:

1. create lower-resolution source and reference products
2. coregister the lower-resolution source to the lower-resolution reference
3. use the coarse result to create an intermediate full-resolution source that is closer to the reference
4. run Coregix again at full resolution to refine the alignment

The first pass handles the larger residual offset on a simpler image pair. The second pass starts from a closer alignment, giving the area-based registration a better chance of converging on the fine-scale correction.

## Grid and Metadata Behavior

Chunked output follows the same grid rules as standard alignment:

- source-grid output is the default
- reference-grid output is selected with `--no-output-on-moving-grid`
- band metadata and descriptions are copied from the source raster when possible
- output nodata defaults to source nodata, then reference nodata, then `0`

Chunking changes how the transform is estimated and applied internally; it does not change the output file format or the public API.
