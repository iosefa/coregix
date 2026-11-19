"""GDAL helpers needed by joint coregistration."""

import math

import numpy as np
from osgeo import gdal

from .handlers import _resolve_paths
from .types_and_validation import Universal, _validate_output_grid
from .utils_logging import _print_step_start, _report_image_result
from .utils_multiprocessing import _get_executor, _resolve_parallel_config, _run_image_tasks
from .utils_output import uses_writing_postfix
from .utils_progress import gdal_progress, reports_progress
from .utils_thread_balance import current_tile_threads


def _find_overlaps(image_bounds_dict):
    overlaps = []
    names = sorted(image_bounds_dict)
    for index, first in enumerate(names):
        left, bottom, right, top = image_bounds_dict[first]
        for second in names[index + 1:]:
            other_left, other_bottom, other_right, other_top = image_bounds_dict[second]
            if (left < other_right and right > other_left
                    and bottom < other_top and top > other_bottom):
                overlaps.append((first, second))
    return tuple(overlaps)


def compute_resolution(paths, strategy):
    _validate_output_grid(resolution=strategy)
    if strategy is None:
        return None
    if isinstance(strategy, (int, float)):
        return float(strategy), float(strategy)
    resolutions = []
    for path in paths:
        dataset = gdal.Open(path, gdal.GA_ReadOnly)
        transform = dataset.GetGeoTransform()
        resolutions.append((math.hypot(transform[1], transform[4]),
                            math.hypot(transform[2], transform[5])))
        dataset = None
    values = np.asarray(resolutions, dtype=float)
    if strategy == "highest":
        return float(values[:, 0].min()), float(values[:, 1].min())
    if strategy == "lowest":
        return float(values[:, 0].max()), float(values[:, 1].max())
    return float(values[:, 0].mean()), float(values[:, 1].mean())


def _set_gdal_cache(cache, debug_logs):
    if cache is not None:
        gdal.SetCacheMax(int(cache * 1024 ** 3))


def _set_gdal_workers(io_threads, debug_logs):
    if io_threads is not None:
        workers = "ALL_CPUS" if io_threads == "cpu" else str(io_threads)
        gdal.SetConfigOption("GDAL_NUM_THREADS", workers)


def _resolve_gdal_dtype(override_dtype=None, input_image_path=None, debug_logs=False):
    if override_dtype is not None:
        return override_dtype
    if input_image_path is None:
        raise ValueError("input_image_path must be provided if override_dtype is None")
    dataset = gdal.Open(input_image_path, gdal.GA_ReadOnly)
    dtype = gdal.GetDataTypeName(dataset.GetRasterBand(1).DataType)
    dataset = None
    return dtype


def _resolve_window_size(window_size, input_image_path, debug_logs):
    if window_size is not None:
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer or None.")
        return int(window_size)
    dataset = gdal.Open(input_image_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"Could not open {input_image_path}")
    block_x, block_y = dataset.GetRasterBand(1).GetBlockSize()
    dataset = None
    if block_x > 1 and block_y > 1:
        return max(block_x, block_y)
    return None


def _get_gdal_bounds(path):
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"Could not open {path}")
    width, height = dataset.RasterXSize, dataset.RasterYSize
    transform = dataset.GetGeoTransform()
    corners = [gdal.ApplyGeoTransform(transform, x, y)
               for x, y in ((0, 0), (width, 0), (width, height), (0, height))]
    dataset = None
    xs, ys = zip(*corners)
    return min(xs), min(ys), max(xs), max(ys)


@reports_progress(worker_progress=True)
@uses_writing_postfix
def compute_overviews(
    input_images_paths,
    *,
    window_scales=(2, 4, 8, 16, 32),
    cache=None,
    image_threads=None,
    io_threads=None,
    tile_threads=None,
    concurrent_processing_backend="process_pool",
    dask_scheduler=None,
    debug_logs=False,
    validity_writing_postfix=None,
):
    _print_step_start("compute_overviews")
    Universal._validate(
        input_images=input_images_paths, window_scales=window_scales,
        cache=cache, image_threads=image_threads, io_threads=io_threads,
        tile_threads=tile_threads, concurrent_processing_backend=concurrent_processing_backend,
        dask_scheduler=dask_scheduler,
    )
    paths = _resolve_paths("search", input_images_paths,
                           kwargs={"default_file_pattern": "*.tif"})
    if not window_scales:
        return paths
    _set_gdal_cache(cache, debug_logs)
    _set_gdal_workers(io_threads, debug_logs)
    parallel, workers = _resolve_parallel_config(
        image_threads, concurrent_processing_backend, dask_scheduler)
    tile_on, tile_workers = _resolve_parallel_config(tile_threads)
    args = [(path, window_scales, tile_on, tile_workers, debug_logs) for path in paths]
    _run_image_tasks(
        _process_image_overview, args, input_paths=paths, output_paths=paths,
        parallel=parallel, backend="thread", workers=workers,
        concurrent_processing_backend=concurrent_processing_backend,
        dask_scheduler=dask_scheduler, executor_factory=_get_executor,
    )
    return paths


def _process_image_overview(path, window_scales, tile_threads_on, tile_workers, debug_logs):
    dataset = gdal.Open(path, gdal.GA_Update)
    if dataset is None:
        raise RuntimeError(f"Cannot open {path}")
    options = [f"NUM_THREADS={current_tile_threads(tile_workers)}"] if tile_threads_on else []
    dataset.BuildOverviews("AVERAGE", window_scales, options=options,
                           callback=gdal_progress("Building overviews"))
    dataset = None
    if debug_logs:
        _report_image_result("Overview levels", len(window_scales))
