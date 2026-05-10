# Edge Trimming

The edge trimming API provides post-processing for invalid border artifacts around nodata regions.

Import from `coregix.postprocess`:

```python
from coregix.postprocess import trim_edge_invalid_pixels, EdgeTrimResult
```

::: coregix.postprocess
    options:
      members:
        - EdgeTrimResult
        - trim_edge_invalid_pixels
