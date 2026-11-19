# Concepts

Tie points link corresponding pixels across overlapping moving rasters and fixed references. ORB detects features in search windows; Elastix estimates translations using mutual information. Search centers use a regular grid or spread across valid overlap pixels.

A sparse joint solve estimates image-wide translation, similarity, or affine corrections. A second solve estimates residual mesh displacements with movement, smoothness, and bending penalties. Fixed references stay stationary. Movement weights control resistance to correction; alignment strengths scale each image's solved correction.

The global and local corrections are composed for output warping. Matching uses raster masks, nodata, and finite values in selected bands. Band numbers are one-based, and all bands are retained in outputs. Inputs must share a CRS and have unique basenames.
