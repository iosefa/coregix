# Edge Trimming

Coregix can trim invalid edge artifacts after alignment. This is useful when interpolation or chunked transforms leave irregular invalid boundaries near nodata regions.

## Integrated Alignment Option

Run trimming as part of alignment with `--trim-edge-invalid`:

```bash
vhr-align-image-pair \
  --moving-image /path/to/moving_large.tif \
  --fixed-image /path/to/fixed.tif \
  --output-image /path/to/aligned_edgefixed.tif \
  --split-factor 2 \
  --trim-edge-invalid \
  --edge-trim-depth 8 \
  --edge-trim-invalid-below -3000
```

Python:

```python
from coregix import align_image_pair

result = align_image_pair(
    moving_image_path="/path/to/moving_large.tif",
    fixed_image_path="/path/to/fixed.tif",
    output_image_path="/path/to/aligned_edgefixed.tif",
    split_factor=2,
    trim_edge_invalid=True,
    edge_trim_depth=8,
    edge_trim_invalid_below=-3000,
)
```

## Standalone Trimming

You can trim an existing aligned raster with the module CLI:

```bash
python -m coregix.cli.trim_edge_invalid \
  --input-image /path/to/aligned.tif \
  --output-image /path/to/aligned_trimmed.tif \
  --edge-depth 8 \
  --invalid-below -3000
```

Or from Python:

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

## Invalid Criteria

The trim pass detects invalid pixels from:

- the raster nodata value
- `invalid_below`
- `invalid_above`

Use threshold options when artifacts are not exactly equal to the dataset nodata value.

## In-place Updates

The standalone tool can modify an input raster in place:

```bash
python -m coregix.cli.trim_edge_invalid \
  --input-image /path/to/aligned.tif \
  --in-place \
  --edge-depth 8
```

Use this only when the source raster can be overwritten.
