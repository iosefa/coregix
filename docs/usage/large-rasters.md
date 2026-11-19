# Large Rasters

Matching reads local windows within image overlaps. `tie_search_radius` controls search-window size, and `tie_max_samples` controls sampling density. `local_grid_spacing` controls deformation mesh density independently.

GDAL applies the output warp. `window_size`, `cache` (gigabytes), `image_threads`, `io_threads`, and `tile_threads` control output tiles, caching, and concurrency:

```python
from coregix import align_image_pair

outputs = align_image_pair(
    ["a.tif", "b.tif"], "aligned/",
    cache=2, image_threads=2, window_size=512, save_as_cog=True,
)
```

Use `tie_save_path` to cache selected ties for repeated solves. Use `resume_from_outputs="validate"` to reuse existing outputs that pass validation.
