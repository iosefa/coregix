"""Pairwise alignment pipeline."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds

from coregix.preprocess.registration import (
    apply_elastix_transform_subprocess,
    estimate_elastix_transform,
    write_transform_parameter_files,
)

DEFAULT_ALIGNMENT_PARAMETER_MAPS = ["translation", "rigid"]


@dataclass
class AlignmentResult:
    """Summary of pairwise alignment execution.

    Attributes:
        output_image_path: Final aligned image path.
        temp_dir: Retained temporary directory path when ``keep_temp_dir=True``; otherwise ``None``.
    """

    output_image_path: str
    temp_dir: Optional[str]


def _to_int_window(window: Window, max_width: int, max_height: int) -> Window:
    col0 = max(0, int(np.floor(window.col_off)))
    row0 = max(0, int(np.floor(window.row_off)))
    col1 = min(max_width, int(np.ceil(window.col_off + window.width)))
    row1 = min(max_height, int(np.ceil(window.row_off + window.height)))
    return Window(col_off=col0, row_off=row0, width=max(0, col1 - col0), height=max(0, row1 - row0))


def _write_single_band_tif(
    path: str,
    data: np.ndarray,
    *,
    crs,
    transform,
    dtype: str,
    nodata: Optional[float],
) -> None:
    profile = {
        "driver": "GTiff",
        "count": 1,
        "height": int(data.shape[0]),
        "width": int(data.shape[1]),
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "bigtiff": "yes",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _make_output_profile(
    src_profile: dict,
    *,
    count: int,
    dtype: str,
    nodata: float,
    width: Optional[int] = None,
    height: Optional[int] = None,
    transform=None,
) -> dict:
    """Build a GeoTIFF output profile without inheriting invalid tiling metadata."""
    profile = src_profile.copy()

    # Source profiles can carry nonstandard block sizes that are valid for reading
    # but rejected when reused as GeoTIFF creation options for a new file.
    for key in ("blockxsize", "blockysize"):
        profile.pop(key, None)

    profile.update(
        count=count,
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        predictor=2,
        tiled=True,
        interleave="band",
        BIGTIFF="IF_SAFER",
    )
    if width is not None:
        profile["width"] = int(width)
    if height is not None:
        profile["height"] = int(height)
    if transform is not None:
        profile["transform"] = transform
    return profile


def _resolve_nodata(src: rasterio.DatasetReader, override_nodata: Optional[float]) -> Optional[float]:
    return override_nodata if override_nodata is not None else src.nodata


def _edge_proxy(data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    arr = data.astype(np.float32, copy=True)
    arr[~valid_mask] = 0.0
    gy, gx = np.gradient(arr)
    edge = np.hypot(gx, gy)
    edge[~valid_mask] = 0.0
    return edge.astype(np.float32)


def align_image_pair(
    moving_image_path: str,
    fixed_image_path: str,
    output_image_path: str,
    *,
    band_index: int = 0,
    moving_band_index: Optional[int] = None,
    fixed_band_index: Optional[int] = None,
    parameter_file_paths: Optional[List[str]] = None,
    moving_nodata: Optional[float] = None,
    fixed_nodata: Optional[float] = None,
    output_nodata: Optional[float] = None,
    min_valid_fraction: float = 0.01,
    temp_dir: Optional[str] = None,
    keep_temp_dir: bool = False,
    log_to_console: bool = False,
    clip_fixed_to_moving: bool = False,
    output_on_moving_grid: bool = True,
    enforce_mutual_valid_mask: bool = False,
    use_edge_proxies: bool = True,
) -> AlignmentResult:
    """Align a moving image onto a fixed image using the core elastix workflow.

    Transforms are estimated over the full fixed-grid ROI using a built-in
    translation->rigid elastix schedule unless explicit parameter files are
    provided. By default registration runs on edge-proxy images; raw intensities
    can be used instead. The resulting transform is then applied to all
    moving-image bands and written to the final output.

    Args:
        moving_image_path: Path to moving image A (image to be warped).
        fixed_image_path: Path to fixed/reference image B (target grid).
        output_image_path: Path for final aligned output image.
        band_index: 0-based band index used for registration metric.
        parameter_file_paths: Optional elastix parameter file path(s). When omitted,
            the built-in translation->rigid schedule is used.
        moving_nodata: Optional moving-image nodata override for mask generation.
        fixed_nodata: Optional fixed-image nodata override for mask generation.
        output_nodata: Optional output nodata override. Defaults to moving nodata,
            then fixed nodata, else ``0``.
        min_valid_fraction: Minimum valid-mask fraction required in the registration ROI.
        temp_dir: Optional parent directory for temporary working artifacts.
        keep_temp_dir: If ``True``, keep the temporary working directory for inspection.
        log_to_console: If ``True``, emit elastix/transformix logs to stdout.
        clip_fixed_to_moving: If ``True``, restrict fixed-image domain to moving-image bounds.
        output_on_moving_grid: If ``True``, write final output on the moving-image grid
            (same transform, size, and pixel size as moving image).
        enforce_mutual_valid_mask: If ``True``, constrain both fixed and moving
            elastix masks to the mutual valid-data overlap of both images.
        use_edge_proxies: If ``True``, register on edge-proxy images rather than
            raw intensities.

    Returns:
        AlignmentResult summary with output path.

    Raises:
        ValueError: If argument values are out of range or images are incompatible.
    """
    if band_index < 0:
        raise ValueError("band_index must be >= 0 (0-based).")
    if moving_band_index is not None and moving_band_index < 0:
        raise ValueError("moving_band_index must be >= 0 (0-based).")
    if fixed_band_index is not None and fixed_band_index < 0:
        raise ValueError("fixed_band_index must be >= 0 (0-based).")
    if min_valid_fraction <= 0 or min_valid_fraction > 1:
        raise ValueError("min_valid_fraction must be in (0, 1].")

    temp_ctx = None
    work_dir: str
    if keep_temp_dir:
        work_dir = tempfile.mkdtemp(prefix="vhr_align_", dir=temp_dir)
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="vhr_align_", dir=temp_dir)
        work_dir = temp_ctx.name

    with rasterio.open(fixed_image_path) as fixed_src, rasterio.open(moving_image_path) as moving_src:
        moving_band_1based = (moving_band_index if moving_band_index is not None else band_index) + 1
        fixed_band_1based = (fixed_band_index if fixed_band_index is not None else band_index) + 1
        if fixed_band_1based > fixed_src.count:
            raise ValueError(
                f"Requested fixed band index={fixed_band_1based - 1}, but fixed image has {fixed_src.count} band(s)."
            )
        if moving_band_1based > moving_src.count:
            raise ValueError(
                f"Requested moving band index={moving_band_1based - 1}, but moving image has {moving_src.count} band(s)."
            )
        if fixed_src.crs is None or moving_src.crs is None:
            raise ValueError("Both fixed and moving images must have CRS.")
        if fixed_src.crs != moving_src.crs:
            raise ValueError(
                "Fixed and moving images must share the same CRS for tile-window extraction. "
                f"fixed={fixed_src.crs}, moving={moving_src.crs}"
            )

        moving_nodata_value = _resolve_nodata(moving_src, moving_nodata)
        fixed_nodata_value = _resolve_nodata(fixed_src, fixed_nodata)
        out_nodata = (
            output_nodata
            if output_nodata is not None
            else moving_nodata_value
            if moving_nodata_value is not None
            else fixed_nodata_value
            if fixed_nodata_value is not None
            else 0
        )

        if clip_fixed_to_moving:
            fixed_domain_window = _to_int_window(
                from_bounds(
                    left=moving_src.bounds.left,
                    bottom=moving_src.bounds.bottom,
                    right=moving_src.bounds.right,
                    top=moving_src.bounds.top,
                    transform=fixed_src.transform,
                ),
                max_width=fixed_src.width,
                max_height=fixed_src.height,
            )
            if fixed_domain_window.width <= 0 or fixed_domain_window.height <= 0:
                raise ValueError(
                    "No overlap between moving-image bounds and fixed-image grid."
                )
        else:
            fixed_domain_window = Window(
                col_off=0, row_off=0, width=fixed_src.width, height=fixed_src.height
            )

        os.makedirs(os.path.dirname(output_image_path) or ".", exist_ok=True)
        if output_on_moving_grid:
            out_profile = _make_output_profile(
                moving_src.profile,
                count=moving_src.count,
                dtype=moving_src.dtypes[0],
                nodata=out_nodata,
            )
        else:
            out_profile = _make_output_profile(
                fixed_src.profile,
                count=moving_src.count,
                dtype=moving_src.dtypes[0],
                nodata=out_nodata,
                width=int(fixed_domain_window.width),
                height=int(fixed_domain_window.height),
                transform=fixed_src.window_transform(fixed_domain_window),
            )

        fixed_window = fixed_domain_window
        core_out_window = Window(
            col_off=0,
            row_off=0,
            width=int(fixed_domain_window.width),
            height=int(fixed_domain_window.height),
        )
        fixed_bounds = fixed_src.window_bounds(fixed_window)
        moving_window = _to_int_window(
            from_bounds(
                left=fixed_bounds[0],
                bottom=fixed_bounds[1],
                right=fixed_bounds[2],
                top=fixed_bounds[3],
                transform=moving_src.transform,
            ),
            max_width=moving_src.width,
            max_height=moving_src.height,
        )
        if moving_window.width <= 0 or moving_window.height <= 0:
            raise ValueError("No overlap between fixed-image ROI and moving-image grid.")

        with rasterio.open(output_image_path, "w+", **out_profile) as out_dst:
            # Preserve radiometric/band metadata from moving image and clear stale stats
            # that can cause misleading display stretches in GIS viewers.
            try:
                out_dst.colorinterp = moving_src.colorinterp
            except Exception:
                pass
            try:
                out_dst.scales = moving_src.scales
                out_dst.offsets = moving_src.offsets
            except Exception:
                pass
            for b in range(1, moving_src.count + 1):
                desc = moving_src.descriptions[b - 1]
                if desc:
                    out_dst.set_band_description(b, desc)
                band_tags = moving_src.tags(b).copy()
                # Remove stale stats tags from source and output; they are often invalid
                # after reprojection/warping and can distort visualization.
                for k in list(band_tags.keys()):
                    if k.upper().startswith("STATISTICS_"):
                        band_tags.pop(k, None)
                out_dst.update_tags(b, **band_tags)
                out_dst.update_tags(
                    b,
                    STATISTICS_MINIMUM="",
                    STATISTICS_MAXIMUM="",
                    STATISTICS_MEAN="",
                    STATISTICS_STDDEV="",
                )

            if output_on_moving_grid:
                # Preserve moving-image pixels outside fixed overlap domain.
                for b in range(1, moving_src.count + 1):
                    for _, block_window in out_dst.block_windows(b):
                        src_block = moving_src.read(b, window=block_window)
                        out_dst.write(src_block.astype(out_profile["dtype"]), b, window=block_window)
            else:
                for b in range(1, moving_src.count + 1):
                    for _, block_window in out_dst.block_windows(b):
                        fill = np.full(
                            (int(block_window.height), int(block_window.width)),
                            out_nodata,
                            dtype=out_profile["dtype"],
                        )
                        out_dst.write(fill, b, window=block_window)

            fixed_band = fixed_src.read(fixed_band_1based, window=fixed_window)
            moving_band = moving_src.read(moving_band_1based, window=moving_window)
            fixed_valid = fixed_src.read_masks(fixed_band_1based, window=fixed_window) > 0
            moving_valid = moving_src.read_masks(moving_band_1based, window=moving_window) > 0
            if fixed_nodata_value is not None:
                fixed_valid &= fixed_band != fixed_nodata_value
            if moving_nodata_value is not None:
                moving_valid &= moving_band != moving_nodata_value

            moving_valid_reprojected = np.zeros(
                (int(fixed_window.height), int(fixed_window.width)),
                dtype=np.uint8,
            )
            reproject(
                source=moving_valid.astype(np.uint8),
                destination=moving_valid_reprojected,
                src_transform=moving_src.window_transform(moving_window),
                src_crs=moving_src.crs,
                dst_transform=fixed_src.window_transform(fixed_window),
                dst_crs=fixed_src.crs,
                src_nodata=0,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )

            fixed_reg_data = fixed_band.astype(np.float32)
            moving_mask_for_elastix: np.ndarray
            fixed_mask_for_elastix: np.ndarray
            moving_on_fixed = np.full(
                (int(fixed_window.height), int(fixed_window.width)),
                moving_nodata_value if moving_nodata_value is not None else out_nodata,
                dtype=np.float32,
            )
            reproject(
                source=moving_band.astype(np.float32),
                destination=moving_on_fixed,
                src_transform=moving_src.window_transform(moving_window),
                src_crs=moving_src.crs,
                dst_transform=fixed_src.window_transform(fixed_window),
                dst_crs=fixed_src.crs,
                src_nodata=moving_nodata_value,
                dst_nodata=moving_nodata_value if moving_nodata_value is not None else out_nodata,
                resampling=Resampling.nearest,
            )
            moving_on_fixed_valid = moving_valid_reprojected > 0
            if use_edge_proxies:
                fixed_reg_data = _edge_proxy(fixed_reg_data, fixed_valid)
                moving_reg_data = _edge_proxy(moving_on_fixed, moving_on_fixed_valid)
                fixed_mask_for_elastix = (fixed_reg_data > 0).astype(np.uint8)
                moving_mask_for_elastix = (moving_reg_data > 0).astype(np.uint8)
            else:
                moving_reg_data = moving_on_fixed
                fixed_mask_for_elastix = fixed_valid.astype(np.uint8)
                moving_mask_for_elastix = moving_on_fixed_valid.astype(np.uint8)
            if enforce_mutual_valid_mask:
                mutual = (fixed_mask_for_elastix > 0) & (moving_mask_for_elastix > 0)
                fixed_mask_for_elastix = mutual.astype(np.uint8)
                moving_mask_for_elastix = mutual.astype(np.uint8)

            min_valid_pixels = int(max(1, min_valid_fraction * (fixed_window.width * fixed_window.height)))
            if int(fixed_mask_for_elastix.sum()) < min_valid_pixels:
                raise ValueError("Insufficient valid fixed-image support in the registration ROI.")
            if int(moving_mask_for_elastix.sum()) < min_valid_pixels:
                raise ValueError("Insufficient valid moving-image support in the registration ROI.")

            fixed_reg_path = os.path.join(work_dir, "fixed_reg.tif")
            moving_reg_path = os.path.join(work_dir, "moving_reg.tif")
            fixed_mask_path = os.path.join(work_dir, "fixed_mask.tif")
            moving_mask_path = os.path.join(work_dir, "moving_mask.tif")

            _write_single_band_tif(
                fixed_reg_path,
                fixed_reg_data.astype("float32"),
                crs=fixed_src.crs,
                transform=fixed_src.window_transform(fixed_window),
                dtype="float32",
                nodata=fixed_nodata_value,
            )
            _write_single_band_tif(
                moving_reg_path,
                moving_reg_data.astype("float32"),
                crs=fixed_src.crs,
                transform=fixed_src.window_transform(fixed_window),
                dtype="float32",
                nodata=moving_nodata_value,
            )
            _write_single_band_tif(
                fixed_mask_path,
                fixed_mask_for_elastix.astype("uint8"),
                crs=fixed_src.crs,
                transform=fixed_src.window_transform(fixed_window),
                dtype="uint8",
                nodata=0,
            )
            _write_single_band_tif(
                moving_mask_path,
                moving_mask_for_elastix.astype("uint8"),
                crs=fixed_src.crs,
                transform=fixed_src.window_transform(fixed_window),
                dtype="uint8",
                nodata=0,
            )

            try:
                transform_parameter_object = estimate_elastix_transform(
                    fixed_image_path=fixed_reg_path,
                    moving_image_path=moving_reg_path,
                    parameter_map=DEFAULT_ALIGNMENT_PARAMETER_MAPS,
                    parameter_file_paths=parameter_file_paths,
                    force_nearest_resample=True,
                    fixed_mask_path=fixed_mask_path,
                    moving_mask_path=moving_mask_path,
                    log_to_console=log_to_console,
                )
            except Exception as exc:
                raise RuntimeError(f"Elastix registration failed: {exc}") from exc

            serialized_transform_files = write_transform_parameter_files(
                transform_parameter_object,
                os.path.join(work_dir, "transform"),
            )

            for b in range(1, moving_src.count + 1):
                moving_band_path = os.path.join(work_dir, f"moving_band_{b:03d}.tif")
                moving_band_data = moving_src.read(b, window=moving_window).astype(np.float32)
                # Registration is estimated on moving data resampled to the fixed-grid
                # ROI. Apply transforms in that same geometry.
                moving_band_on_fixed = np.full(
                    (int(fixed_window.height), int(fixed_window.width)),
                    out_nodata,
                    dtype=np.float32,
                )
                reproject(
                    source=moving_band_data,
                    destination=moving_band_on_fixed,
                    src_transform=moving_src.window_transform(moving_window),
                    src_crs=moving_src.crs,
                    dst_transform=fixed_src.window_transform(fixed_window),
                    dst_crs=fixed_src.crs,
                    src_nodata=moving_nodata_value,
                    dst_nodata=out_nodata,
                    resampling=Resampling.nearest,
                )
                _write_single_band_tif(
                    moving_band_path,
                    moving_band_on_fixed.astype("float32"),
                    crs=fixed_src.crs,
                    transform=fixed_src.window_transform(fixed_window),
                    dtype="float32",
                    nodata=moving_nodata_value,
                )

                transformed_band_path = os.path.join(work_dir, f"warped_band_{b:03d}.tif")
                apply_elastix_transform_subprocess(
                    moving_image_path=moving_band_path,
                    output_image_path=transformed_band_path,
                    parameter_files=serialized_transform_files,
                    reference_image_path=fixed_reg_path,
                    log_to_console=log_to_console,
                )
                with rasterio.open(transformed_band_path) as warped_src:
                    warped_full = warped_src.read(1)
                if output_on_moving_grid:
                    with rasterio.open(transformed_band_path) as warped_src:
                        src_transform = warped_src.transform
                        src_crs = warped_src.crs or fixed_src.crs
                        aligned_window_transform = moving_src.window_transform(moving_window)
                        block_width, block_height = out_dst.block_shapes[b - 1]
                        for row_off in range(0, int(moving_window.height), int(block_height)):
                            for col_off in range(0, int(moving_window.width), int(block_width)):
                                win_w = min(int(block_width), int(moving_window.width) - col_off)
                                win_h = min(int(block_height), int(moving_window.height) - row_off)
                                block_window = Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)
                                remapped_block = np.full(
                                    (win_h, win_w),
                                    out_nodata,
                                    dtype=np.float32,
                                )
                                reproject(
                                    source=rasterio.band(warped_src, 1),
                                    destination=remapped_block,
                                    src_transform=src_transform,
                                    src_crs=src_crs,
                                    src_nodata=out_nodata,
                                    dst_transform=rasterio.windows.transform(block_window, aligned_window_transform),
                                    dst_crs=moving_src.crs,
                                    dst_nodata=out_nodata,
                                    resampling=Resampling.bilinear,
                                )
                                source_block_window = Window(
                                    col_off=int(moving_window.col_off + col_off),
                                    row_off=int(moving_window.row_off + row_off),
                                    width=win_w,
                                    height=win_h,
                                )
                                existing_block = moving_src.read(
                                    b,
                                    window=source_block_window,
                                ).astype(np.float32)
                                valid = remapped_block != out_nodata
                                combined = np.where(valid, remapped_block, existing_block)
                                out_dst.write(
                                    combined.astype(out_profile["dtype"]),
                                    b,
                                    window=source_block_window,
                                )
                else:
                    out_dst.write(warped_full.astype(out_profile["dtype"]), b, window=core_out_window)

    if temp_ctx is not None:
        temp_ctx.cleanup()
        kept_temp = None
    else:
        kept_temp = work_dir

    return AlignmentResult(
        output_image_path=output_image_path,
        temp_dir=kept_temp,
    )


__all__ = ["AlignmentResult", "align_image_pair"]
