# Joint Alignment

Use `align_image_pair` to jointly align overlapping rasters, optionally anchored to fixed references.

```python
from coregix import align_image_pair

outputs = align_image_pair(
    ["source.tif"],
    ["aligned.tif"],
    fixed_images=["reference.tif"],
    input_images_bands=[1],
    fixed_images_bands=[1],
)
```

`input_images` contains moving rasters. `fixed_images` supplies references held stationary in both solvers; these have no output entries. Band numbers are one-based. Outputs retain all bands.

```bash
align-image-pair --input-images source.tif --output-images aligned.tif --fixed-images reference.tif --options '{"local_model":"none"}'
```

`--options` accepts a JSON object of algorithm keyword arguments. The CLI prints a JSON object with an `output_images` list.

Moving pairs default to ORB; reference pairs default to Elastix. Global models are `none`, `translation`, `similarity`, and `affine`. Local models are `none`, `bilinear`, and `piecewise_affine`.

`tie_sampling_method="overlap_spread"` distributes search centers through valid overlap pixels and is the default. `tie_sampling_method="grid"` uses `tie_grid_spacing`. Save selected matches with `tie_save_path` and reuse them with `tie_load_path`; use `tie_save_crs_path` for GIS point export.

`resolution=None` preserves native pixel spacing. Set a positive resolution or use `highest`, `average`, or `lowest` for a shared output resolution. `tap=True` aligns output bounds to the target pixel grid.

The previous pairwise arguments and `AlignmentResult` return type have been removed. The return value is a list of output paths. Use the standalone [edge trimming](edge-trimming.md) API after alignment when needed.
