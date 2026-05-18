# Concepts

Coregix performs pairwise raster coregistration. A source raster is aligned to a reference raster, and the result is written as a georeferenced GeoTIFF. The public CLI uses `--moving-image` for the source raster and `--fixed-image` for the reference raster, following the naming convention used by the registration backend.

## Area-based Registration

Coregix uses area-based image registration. Instead of detecting and matching individual tie points, it compares image values over the overlapping raster area and searches for a transform that improves the similarity between the source and reference images.

The default registration schedule estimates:

1. a translation transform
2. a rigid transform

The rigid stage allows rotation in addition to translation. The default similarity metric is mutual information, which is useful when corresponding surfaces have different radiometry or come from different sensors. This makes the method less dependent on matching raw pixel values exactly, but it still requires shared spatial structure in the images.

## Source and Reference Rasters

The reference raster defines the image that the source raster is coregistered against. The source raster is the image that gets transformed and written to the output.

Coregix expects the source and reference rasters to:

- be in the same CRS
- have overlapping spatial extent
- be close enough for translation/rigid registration to converge
- contain enough valid pixels in the overlap region

Coregix does not perform CRS reprojection as a substitute for registration. If the rasters are in different coordinate reference systems, reproject them before running Coregix.

## Registration Bands

Registration is estimated from one band in each raster. By default, Coregix uses band index `0` from both rasters. Separate source and reference registration bands can be selected when the strongest registration signal is not in the same band position.

```bash
align-image-pair \
  --moving-image source.tif \
  --fixed-image reference.tif \
  --output-image aligned.tif \
  --moving-band-index 0 \
  --fixed-band-index 3
```

All source bands are transformed after the registration model is estimated.

## Edge-proxy Images

By default, Coregix estimates registration on edge-proxy images rather than raw source and reference values. The edge proxy is computed from local image gradients over valid pixels. This emphasizes spatial boundaries and texture while reducing dependence on the original intensity scale.

Edge proxies are useful when the source and reference rasters represent similar ground structure but have different spectral response, acquisition conditions, or value ranges.

Use raw values instead with:

```bash
align-image-pair ... --no-use-edge-proxies
```

or in Python:

```python
align_image_pair(..., use_edge_proxies=False)
```

## Masks and Nodata

Registration should be estimated from valid image support only. Coregix builds masks from each raster's internal mask and nodata metadata, then passes those masks to the registration step.

Override nodata values when the raster metadata is missing or incorrect:

```bash
align-image-pair \
  --moving-image source.tif \
  --fixed-image reference.tif \
  --output-image aligned.tif \
  --moving-nodata 0 \
  --fixed-nodata 0
```

`--min-valid-fraction` controls the minimum valid-pixel fraction required in the registration solve. If the valid area is too small, Coregix stops instead of estimating a poorly constrained transform.

## Solve Grid

Coregix estimates the transform on a solve grid derived from the reference raster. By default, the solve grid uses the reference raster's pixel spacing over the registration region.

`--solve-resolutions` can be used to run one or more solves from coarse to fine, with each pixel size expressed in raster CRS units. Use `0` for the native/reference solve resolution:

```bash
align-image-pair \
  --moving-image source.tif \
  --fixed-image reference.tif \
  --output-image aligned.tif \
  --split-factor 2 \
  --solve-resolutions 8,4,0
```

Each pass refines the previous transform, and the final output is resampled once from the original source raster. During solve-grid preparation, source and reference registration bands and masks are resampled onto the solve grid with nearest-neighbor resampling. This preserves mask classes and avoids creating interpolated values in the registration inputs.

`--solve-resolution` is deprecated and remains available only for single-pass runs. Prefer `--solve-resolutions`, even for one solve, for example `--solve-resolutions 4`.

## Transform Application and Resampling

After the transform is estimated, Coregix applies it to every band in the source raster. The final output is produced by sampling source pixels at transformed coordinates and writing the sampled values to the output grid.

For source-grid output and chunked output, Coregix uses bilinear sampling for raster values and masks invalid samples to nodata. Reference-grid output can use the registration backend's resampling path when the solve grid matches the reference grid, and bilinear resampling when remapping from a coarser solve grid. In general, Coregix outputs are best suited to continuous image data. For categorical rasters, class labels may be mixed by interpolation; those products should be handled with care.

Output nodata defaults to the source nodata value, then the reference nodata value, then `0` if neither raster declares nodata. It can be set explicitly with `--output-nodata`.

## Output Grid

By default, Coregix writes the result on the source-raster grid. This preserves the source raster's dimensions, transform, band count, and most band metadata, while replacing pixels in the registered overlap region.

```bash
align-image-pair \
  --moving-image source.tif \
  --fixed-image reference.tif \
  --output-image aligned.tif \
  --output-on-moving-grid
```

Write on the reference-raster grid with:

```bash
align-image-pair \
  --moving-image source.tif \
  --fixed-image reference.tif \
  --output-image aligned_on_reference_grid.tif \
  --no-output-on-moving-grid
```

Use the source grid when the output needs to remain compatible with an existing source stack. Use the reference grid when preparing directly comparable raster pairs.

## Chunking Large Rasters

`split_factor` enables chunked processing for large rasters. The number of chunks is `2 ** split_factor`:

| `split_factor` | Chunks |
| --- | --- |
| `0` | no split |
| `1` | 2 chunks |
| `2` | 4 chunks |
| `3` | 8 chunks |

For chunked runs, Coregix estimates local registrations over overlapping chunks, samples anchor correspondences from successful chunks, and fits a single global rigid transform from those correspondences. The final output is then written block by block. This reduces memory pressure while keeping the output model globally consistent.

Use the smallest `split_factor` that keeps processing stable. Higher values increase overhead and can fail if individual chunks do not contain enough valid, informative overlap.

## Edge Cleanup

Resampling can leave invalid border artifacts around nodata regions. `--trim-edge-invalid` applies a post-processing pass that detects invalid areas and expands them by a configurable number of pixels.

```bash
align-image-pair \
  --moving-image source.tif \
  --fixed-image reference.tif \
  --output-image aligned_trimmed.tif \
  --trim-edge-invalid \
  --edge-trim-depth 8
```

Thresholds such as `--edge-trim-invalid-below` and `--edge-trim-invalid-above` are available when artifacts are not exactly equal to the raster nodata value.

## Practical Limits

Coregix is best suited to residual geometric offsets between already georeferenced rasters. It is not intended to solve:

- large CRS or datum mismatches
- severe terrain or sensor-model errors
- nonrigid deformation across the scene
- rasters with little spatial overlap
- rasters with weak shared spatial structure

For those cases, correct the georeferencing first or use a registration model designed for local deformation.
