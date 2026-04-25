"""Image registration utilities."""

from typing import Any, Optional, Sequence, Union

import numpy as np


def _require_itk():
    try:
        import itk
    except ImportError as exc:
        raise RuntimeError(
            "itk-elastix is not installed. Install extras with `pip install -e \".[elastix]\"`."
        ) from exc
    return itk


def estimate_elastix_transform(
    fixed_image_path: str,
    moving_image_path: str,
    *,
    parameter_map: Union[str, Sequence[str]] = "rigid",
    force_linear_resample: bool = False,
    force_nearest_resample: bool = False,
    fixed_mask_path: Optional[str] = None,
    moving_mask_path: Optional[str] = None,
    log_to_console: bool = False,
) -> Any:
    """Estimate transform parameters that map a moving image to a fixed image.

    Args:
        fixed_image_path: Path to the fixed/reference image.
        moving_image_path: Path to the moving image to be aligned.
        parameter_map: Elastix parameter map name(s). Typical values include
            ``"translation"``, ``"rigid"``, ``"affine"``, and ``"bspline"``.
        force_linear_resample: If ``True``, enforce linear final resampling in all
            loaded parameter maps (mimics explicit wrapper settings).
        force_nearest_resample: If ``True``, enforce nearest-neighbor final
            resampling in all loaded parameter maps.
        fixed_mask_path: Optional fixed-image mask path.
        moving_mask_path: Optional moving-image mask path.
        log_to_console: If ``True``, emit elastix logs to stdout.

    Returns:
        ITK transform parameter object produced by elastix.

    Raises:
        RuntimeError: If ``itk-elastix`` is not installed.
    """
    itk = _require_itk()

    fixed = itk.imread(fixed_image_path, itk.F)
    moving = itk.imread(moving_image_path, itk.F)

    parameter_object = itk.ParameterObject.New()
    map_names = [parameter_map] if isinstance(parameter_map, str) else list(parameter_map)
    for map_name in map_names:
        parameter_object.AddParameterMap(parameter_object.GetDefaultParameterMap(map_name))
    if force_linear_resample:
        for idx in range(int(parameter_object.GetNumberOfParameterMaps())):
            parameter_object.SetParameter(idx, "ResampleInterpolator", "FinalLinearInterpolator")
            parameter_object.SetParameter(idx, "DefaultPixelValue", "0")
            parameter_object.SetParameter(idx, "FinalBSplineInterpolationOrder", "1")
    if force_nearest_resample:
        for idx in range(int(parameter_object.GetNumberOfParameterMaps())):
            parameter_object.SetParameter(idx, "ResampleInterpolator", "FinalNearestNeighborInterpolator")
            parameter_object.SetParameter(idx, "DefaultPixelValue", "0")
            parameter_object.SetParameter(idx, "FinalBSplineInterpolationOrder", "0")

    kwargs = {
        "fixed_image": fixed,
        "moving_image": moving,
        "parameter_object": parameter_object,
        "log_to_console": log_to_console,
    }

    if fixed_mask_path:
        kwargs["fixed_mask"] = itk.imread(fixed_mask_path, itk.UC)
    if moving_mask_path:
        kwargs["moving_mask"] = itk.imread(moving_mask_path, itk.UC)

    _, transform_parameter_object = itk.elastix_registration_method(**kwargs)
    return transform_parameter_object


def apply_elastix_transform(
    moving_image_path: str,
    output_image_path: str,
    transform_parameter_object: Any,
    *,
    reference_image_path: Optional[str] = None,
    log_to_console: bool = False,
) -> str:
    """Apply a precomputed elastix transform to an image and write the result.

    Args:
        moving_image_path: Path to the moving image to warp.
        output_image_path: Output path for the transformed image.
        transform_parameter_object: Transform parameter object from
            :func:`estimate_elastix_transform`.
        reference_image_path: Optional raster whose CRS/transform are copied to
            ``output_image_path`` after transformix writes the data.
        log_to_console: If ``True``, emit transformix logs to stdout.

    Returns:
        The written ``output_image_path``.

    Raises:
        RuntimeError: If ``itk-elastix`` is not installed.
    """
    itk = _require_itk()
    moving = itk.imread(moving_image_path, itk.F)
    transformed = itk.transformix_filter(
        moving,
        transform_parameter_object=transform_parameter_object,
        log_to_console=log_to_console,
    )
    itk.imwrite(transformed, output_image_path)
    if reference_image_path:
        import rasterio

        with rasterio.open(reference_image_path) as ref, rasterio.open(output_image_path, "r+") as out:
            out.crs = ref.crs
            out.transform = ref.transform
            if ref.nodata is not None:
                out.nodata = ref.nodata
    return output_image_path


def apply_elastix_transform_array(
    moving_image: np.ndarray,
    transform_parameter_object: Any,
    *,
    log_to_console: bool = False,
) -> np.ndarray:
    """Apply a precomputed elastix transform to an in-memory image array."""
    itk = _require_itk()
    moving = itk.image_from_array(np.asarray(moving_image, dtype=np.float32))
    transformed = itk.transformix_filter(
        moving,
        transform_parameter_object=transform_parameter_object,
        log_to_console=log_to_console,
    )
    return itk.array_from_image(transformed)


def deformation_field_from_transform(
    reference_image_path: str,
    transform_parameter_object: Any,
    *,
    output_directory: Optional[str] = None,
) -> np.ndarray:
    """Return the transformix deformation field as a NumPy array."""
    itk = _require_itk()
    reference = itk.imread(reference_image_path, itk.F)
    kwargs = {
        "transform_parameter_object": transform_parameter_object,
    }
    if output_directory is not None:
        kwargs["output_directory"] = output_directory
    field = itk.transformix_deformation_field(reference, **kwargs)
    return itk.array_from_image(field)


def deformation_field_from_transform_region(
    transform_parameter_object: Any,
    *,
    row_off: int,
    col_off: int,
    height: int,
    width: int,
    output_directory: Optional[str] = None,
) -> np.ndarray:
    """Return a deformation field for a fixed-grid subregion."""
    itk = _require_itk()
    reference = itk.image_from_array(np.zeros((height, width), dtype=np.float32))
    reference.SetOrigin((float(col_off), float(row_off)))
    reference.SetSpacing((1.0, 1.0))
    kwargs = {
        "transform_parameter_object": transform_parameter_object,
    }
    if output_directory is not None:
        kwargs["output_directory"] = output_directory
    field = itk.transformix_deformation_field(reference, **kwargs)
    return itk.array_from_image(field)


def run_elastix_registration(
    fixed_image_path: str,
    moving_image_path: str,
    output_image_path: str,
    *,
    parameter_map: Union[str, Sequence[str]] = "rigid",
    fixed_mask_path: Optional[str] = None,
    moving_mask_path: Optional[str] = None,
    log_to_console: bool = False,
) -> str:
    """Register a moving image to a fixed image using itk-elastix.

    Args:
        fixed_image_path: Path to the fixed/reference image.
        moving_image_path: Path to the moving image that will be warped.
        output_image_path: Path where the registered image will be written.
        parameter_map: Elastix parameter map name(s). Common values:
            ``"rigid"``, ``"affine"``, ``"bspline"``.
        fixed_mask_path: Optional fixed-image mask path.
        moving_mask_path: Optional moving-image mask path.
        log_to_console: Whether elastix should print logs to stdout.

    Returns:
        The written ``output_image_path``.
    """
    transform_parameter_object = estimate_elastix_transform(
        fixed_image_path=fixed_image_path,
        moving_image_path=moving_image_path,
        parameter_map=parameter_map,
        fixed_mask_path=fixed_mask_path,
        moving_mask_path=moving_mask_path,
        log_to_console=log_to_console,
    )
    apply_elastix_transform(
        moving_image_path=moving_image_path,
        output_image_path=output_image_path,
        transform_parameter_object=transform_parameter_object,
        log_to_console=log_to_console,
    )
    return output_image_path


__all__ = [
    "estimate_elastix_transform",
    "apply_elastix_transform",
    "apply_elastix_transform_array",
    "deformation_field_from_transform",
    "deformation_field_from_transform_region",
    "run_elastix_registration",
]
