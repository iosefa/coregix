# API Reference

Coregix exposes a small public API for common use, with lower-level modules available when a project needs more control.

Most users should start with:

```python
from coregix import align_image_pair
```

Use the post-processing API when trimming edge artifacts from an already aligned raster:

```python
from coregix.postprocess import trim_edge_invalid_pixels
```

The registration utilities under `coregix.preprocess.registration` are lower-level wrappers around the registration backend. They are useful for advanced experiments, but they are not required for the standard pairwise raster coregistration path.

## Sections

- [Alignment](alignment.md): primary pairwise raster coregistration API.
- [Edge Trimming](edge-trimming.md): post-processing API for invalid border artifacts.
- [Advanced Registration](registration.md): lower-level transform estimation and application utilities.
