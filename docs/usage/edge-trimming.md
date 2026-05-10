# Edge Trimming

Resampling can leave invalid border artifacts around nodata regions. Edge trimming is an optional cleanup step that detects invalid pixels and expands those invalid regions by a configurable number of pixels.

Use edge trimming when the aligned output contains narrow border artifacts near nodata areas. It is not a replacement for registration; it only modifies output pixels after alignment.

## Trim During Alignment

Add `--trim-edge-invalid` to run cleanup after the coregistered output is written:

```bash
vhr-align-image-pair \
  --moving-image /path/to/source_large.tif \
  --fixed-image /path/to/reference.tif \
  --output-image /path/to/aligned_trimmed.tif \
  --split-factor 2 \
  --trim-edge-invalid \
  --edge-trim-depth 8 \
  --edge-trim-invalid-below -3000
```

Python:

```python
from coregix import align_image_pair

result = align_image_pair(
    moving_image_path="/path/to/source_large.tif",
    fixed_image_path="/path/to/reference.tif",
    output_image_path="/path/to/aligned_trimmed.tif",
    split_factor=2,
    trim_edge_invalid=True,
    edge_trim_depth=8,
    edge_trim_invalid_below=-3000,
)
```

## Trim An Existing Raster

The standalone trim command can be used after alignment:

```bash
python -m coregix.cli.trim_edge_invalid \
  --input-image /path/to/aligned.tif \
  --output-image /path/to/aligned_trimmed.tif \
  --edge-depth 8 \
  --invalid-below -3000
```

Python:

```python
from coregix.postprocess import trim_edge_invalid_pixels

result = trim_edge_invalid_pixels(
    input_image_path="/path/to/aligned.tif",
    output_image_path="/path/to/aligned_trimmed.tif",
    edge_depth=8,
    invalid_below=-3000,
)

print(result.pixels_trimmed)
```

## Invalid Pixel Criteria

The trim pass identifies invalid pixels from:

- the raster nodata value
- `invalid_below`
- `invalid_above`

Thresholds are useful when interpolation artifacts are not exactly equal to the raster's nodata value. For example, `--invalid-below -3000` treats all values at or below `-3000` as invalid while building the trim mask.

## Detection Band

By default, edge artifacts are detected from band index `0`. Select another band when invalid artifacts are more visible elsewhere:

```bash
python -m coregix.cli.trim_edge_invalid \
  --input-image /path/to/aligned.tif \
  --output-image /path/to/aligned_trimmed.tif \
  --detection-band-index 2 \
  --edge-depth 8
```

The resulting trim mask is applied to all bands.

## In-place Updates

The standalone tool can modify an input raster in place:

```bash
python -m coregix.cli.trim_edge_invalid \
  --input-image /path/to/aligned.tif \
  --in-place \
  --edge-depth 8
```

Use in-place updates only when the input raster can be overwritten.
