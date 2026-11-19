import json
import math
import re
from typing import Tuple, List, Literal, Optional

_UNSET = object()


# Universal types
class Universal:
    SearchFolderOrListFiles = str | List[str]
    CreateInFolderOrListFiles = str | List[str]
    SaveAsCog = bool  # Default: True
    DebugLogs = bool  # Default: False
    VectorMask = Tuple[Literal["include", "exclude"], str, Optional[str]] | None
    WindowSize = int | None
    CustomNodataValue = float | int | None
    Threads = Literal["cpu"] | int | None
    ConcurrentProcessingBackend = Literal["process_pool", "dask"]
    DaskScheduler = Tuple[Literal["file", "address"], str] | None
    Cache = float | None
    CalculationDtype = str
    CustomOutputDtype = str | None
    CreateNameAttribute: Tuple[str, str] | None
    Resolution = Literal["highest", "average", "lowest"] | int | float | None

    @staticmethod
    def _validate(
        *,
        input_images=_UNSET,
        output_images=_UNSET,
        save_as_cog=_UNSET,
        debug_logs=_UNSET,
        vector_mask=_UNSET,
        window_size=_UNSET,
        window_scales=_UNSET,
        custom_nodata_value=_UNSET,
        calculation_dtype=_UNSET,
        custom_output_dtype=_UNSET,
        create_name_attribute=_UNSET,
        output_dtype=_UNSET,
        cache=_UNSET,
        image_threads=_UNSET,
        io_threads=_UNSET,
        tile_threads=_UNSET,
        estimate_stats=_UNSET,
        concurrent_processing_backend=_UNSET,
        dask_scheduler=_UNSET,
    ):
        if input_images is not _UNSET:
            if not isinstance(input_images, (str, list)):
                raise ValueError(
                    "input_images must be a string (path or glob pattern) or a list of strings."
                )
            if isinstance(input_images, list) and not all(
                isinstance(p, str) for p in input_images
            ):
                raise ValueError("All elements in input_images list must be strings.")

        if output_images is not _UNSET:
            if not isinstance(output_images, (str, list)):
                raise ValueError(
                    "output_images must be a string (path or template) or a list of strings."
                )
            if isinstance(output_images, list) and not all(
                isinstance(p, str) for p in output_images
            ):
                raise ValueError("All elements in output_images list must be strings.")

        if save_as_cog is not _UNSET:
            if not isinstance(save_as_cog, bool):
                raise ValueError("save_as_cog must be a boolean.")

            if save_as_cog:
                if window_size is _UNSET or window_size is None:
                    raise ValueError("When save_as_cog=True, window_size must be set.")
                if window_size % 16 != 0:
                    raise ValueError("When save_as_cog=True, window_size must be a multiple of 16.")

        if debug_logs is not _UNSET:
            if not isinstance(debug_logs, bool):
                raise ValueError("debug_logs must be a boolean.")

        if vector_mask is not _UNSET and vector_mask is not None:
            if not isinstance(vector_mask, tuple) or len(vector_mask) not in {2, 3}:
                raise ValueError("vector_mask must be a tuple of 2 or 3 elements.")
            if vector_mask[0] not in {"include", "exclude"}:
                raise ValueError(
                    "The first element of vector_mask must be 'include' or 'exclude'."
                )
            if not isinstance(vector_mask[1], str):
                raise ValueError(
                    "The second element must be a string (vector file path)."
                )
            if len(vector_mask) == 3 and not isinstance(vector_mask[2], str):
                raise ValueError(
                    "The third element, if provided, must be a string (field name)."
                )

        if window_size is not _UNSET:
            if window_size is None:
                pass
            elif isinstance(window_size, int):
                if window_size <= 0:
                    raise ValueError("window_size must be > 0.")
            else:
                raise ValueError("window_size must be an int or None.")

        if window_scales is not _UNSET and window_scales is not None:
            if (not isinstance(window_scales, tuple)
                    or not all(type(scale) is int and scale > 1 for scale in window_scales)
                    or tuple(sorted(set(window_scales))) != window_scales):
                raise ValueError("window_scales must be an increasing tuple of unique integers greater than 1, or None.")

        if custom_nodata_value is not _UNSET:
            if custom_nodata_value is not None and not isinstance(
                custom_nodata_value, (int, float)
            ):
                raise ValueError("custom_nodata_value must be a number or None.")

        if calculation_dtype is not _UNSET:
            if not isinstance(calculation_dtype, str):
                raise ValueError("calculation_dtype must be a string.")

        if custom_output_dtype is not _UNSET and custom_output_dtype is not None:
            if not isinstance(custom_output_dtype, str):
                raise ValueError("custom_output_dtype must be a string or None.")

        if create_name_attribute is not _UNSET:
            if (
                not isinstance(Universal.CreateNameAttribute, tuple)
                or len(Universal.CreateNameAttribute) != 2
            ):
                raise ValueError(
                    "CreateNameAttribute must be a tuple of two strings or None."
                )
            if not all(isinstance(s, str) for s in Universal.CreateNameAttribute):
                raise ValueError(
                    "Both elements of CreateNameAttribute must be strings."
                )
        if output_dtype is not _UNSET and output_dtype is not None:
            if not isinstance(output_dtype, str):
                raise ValueError("output_dtype must be a string or None.")

        if cache is not _UNSET:
            if cache is None:
                pass
            elif isinstance(cache, (int, float)) and not isinstance(cache, bool):
                if cache <= 0:
                    raise ValueError("cache must be > 0 (in GB).")
            else:
                raise ValueError("cache must be a number in GB, or None.")

        if image_threads is not _UNSET:
            _validate_threads(image_threads, "image_threads")

        if io_threads is not _UNSET:
            _validate_threads(io_threads, "io_threads")

        if tile_threads is not _UNSET:
            _validate_threads(tile_threads, "tile_threads")

        if estimate_stats is not _UNSET:
            if not isinstance(estimate_stats, bool):
                raise ValueError("estimate_stats must be a boolean.")

        if concurrent_processing_backend is not _UNSET:
            if concurrent_processing_backend not in {"process_pool", "dask"}:
                raise ValueError(
                    'concurrent_processing_backend must be "process_pool" or "dask".'
                )

        if dask_scheduler is not _UNSET:
            _validate_dask_scheduler(dask_scheduler)

        if concurrent_processing_backend is not _UNSET and dask_scheduler is not _UNSET:
            _validate_concurrent_processing_config(
                concurrent_processing_backend,
                dask_scheduler,
                None if image_threads is _UNSET else image_threads,
            )

def _validate_threads(x, name):
    if x is None:
        return
    if isinstance(x, int):
        if x < 1:
            raise ValueError(f"{name} must be a positive integer, got {x}.")
        return
    if isinstance(x, str) and x == "cpu":
        return
    raise ValueError(f'{name} must be "cpu", an int, or None.')


def _validate_dask_scheduler(value):
    if value is None:
        return
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(
            'dask_scheduler must be ("file", path), ("address", address), or None.'
        )
    kind, target = value
    if kind not in {"file", "address"}:
        raise ValueError('dask_scheduler mode must be "file" or "address".')
    if not isinstance(target, str) or not target.strip():
        raise ValueError("dask_scheduler target must be a non-empty string.")


def _validate_concurrent_processing_config(backend, scheduler, image_threads):
    if backend == "dask":
        if scheduler is None:
            raise ValueError(
                'concurrent_processing_backend="dask" requires dask_scheduler.'
            )
        if image_threads is not None:
            raise ValueError(
                'image_threads must be None when concurrent_processing_backend="dask"; '
                "Dask worker capacity is configured on the cluster."
            )
    elif scheduler is not None:
        raise ValueError(
            'dask_scheduler requires concurrent_processing_backend="dask".'
        )


def _validate_output_grid(*, tap=_UNSET, resolution=_UNSET):
    if tap is not _UNSET and not isinstance(tap, bool):
        raise ValueError("tap must be a boolean.")
    if resolution is _UNSET or resolution is None:
        return
    if isinstance(resolution, str) and resolution in {"highest", "average", "lowest"}:
        return
    if (
        isinstance(resolution, (int, float))
        and not isinstance(resolution, bool)
        and math.isfinite(resolution)
        and resolution > 0
    ):
        return
    raise ValueError(
        "resolution must be 'highest', 'average', 'lowest', a positive int or float, or None."
    )


class JointCoregistration:
    @staticmethod
    def _parse_alignment_strength(value: float | str, name: str) -> float | dict[str, float]:
        """Validate a shared strength or an ordered JSON object of pattern-to-strength rules."""
        message = f"{name} must be a number from 0 to 1 or a JSON object string mapping glob patterns to numbers from 0 to 1."
        if isinstance(value, str):
            try:
                rules = json.loads(value)
            except ValueError as error:
                raise ValueError(message) from error
            if not isinstance(rules, dict) or not all(isinstance(pattern, str) for pattern in rules):
                raise ValueError(message)
            values = rules.values()
        else:
            rules = None
            values = (value,)
        if not all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item) and 0 <= item <= 1 for item in values):
            raise ValueError(message)
        return {pattern: float(item) for pattern, item in rules.items()} if rules is not None else float(value)

    @staticmethod
    def _parse_search_radius(value: str) -> tuple[float, str]:
        message = (
            "tie_search_radius must be a positive finite number followed by "
            "'crs' or 'px', for example '100crs' or '128px'."
        )
        match = re.fullmatch(
            r"((?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)(crs|px)",
            value.strip(),
        ) if isinstance(value, str) else None
        if match is None:
            raise ValueError(message)
        magnitude = float(match.group(1))
        if not math.isfinite(magnitude) or magnitude <= 0:
            raise ValueError(message)
        return magnitude, match.group(2)

    @staticmethod
    def _validate(
        *,
        moving_tie_method,
        tie_grid_spacing,
        tie_search_radius,
        tie_orb_max_features,
        tie_max_matches_per_window,
        tie_maximum_displacement,
        tie_ransac_reprojection_threshold,
        tie_robust_loss,
        tie_robust_loss_scale,
        tie_save_path,
        tie_save_crs_path,
        tie_load_path,
        global_model,
        global_image_movement_penalty_weights,
        global_tie_alignment_strength,
        local_model,
        local_image_movement_penalty_weights,
        local_tie_alignment_strength,
        local_grid_spacing,
        local_smoothness_weight,
        local_bending_weight,
        local_anchor_falloff_distance,
        resampling_method,
        tap,
        resolution,
        build_overviews,
        resume_from_outputs,
        fixed_tie_method="elastix",
        fixed_on_no_or_partial_overlap="partial_influence",
        input_images_bands=[1],
        fixed_images_bands=[1],
        tie_sampling_method="overlap_spread",
        tie_max_samples="500crs", tie_min_samples="1000crs", tie_max_attempts=None,
    ):
        from .overlap_sampling import validate_sampling
        validate_sampling(tie_sampling_method, tie_max_samples, tie_min_samples, tie_max_attempts)
        for name, bands in (("input_images_bands", input_images_bands), ("fixed_images_bands", fixed_images_bands)):
            if (not isinstance(bands, list) or not bands
                    or any(isinstance(band, bool) or not isinstance(band, int) or band < 1 for band in bands)
                    or len(set(bands)) != len(bands)):
                raise ValueError(f"{name} must be a nonempty list of unique one-based positive integer band numbers.")
        if fixed_on_no_or_partial_overlap not in {"error", "no_influence", "partial_influence"}:
            raise ValueError("fixed_on_no_or_partial_overlap must be error, no_influence, or partial_influence.")
        if global_model not in {"none", "translation", "similarity", "affine"}:
            raise ValueError("global_model must be 'none', 'translation', 'similarity', or 'affine'.")
        if local_model not in {"none", "bilinear", "piecewise_affine"}:
            raise ValueError("local_model must be 'none', 'bilinear', or 'piecewise_affine'.")
        for name, value in (
            ("global_tie_alignment_strength", global_tie_alignment_strength),
            ("local_tie_alignment_strength", local_tie_alignment_strength),
        ):
            JointCoregistration._parse_alignment_strength(value, name)
        for name, weights in (
            ("global_image_movement_penalty_weights", global_image_movement_penalty_weights),
            ("local_image_movement_penalty_weights", local_image_movement_penalty_weights),
        ):
            if weights is not None and (
                not isinstance(weights, dict)
                or not all(isinstance(key, str) for key in weights)
                or not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and value > 0
                    for value in weights.values()
                )
            ):
                raise ValueError(f"{name} must be a dictionary of glob-pattern: positive-number values or None.")
        for name, value, allow_zero in (
            ("local_grid_spacing", local_grid_spacing, False),
            ("tie_grid_spacing", tie_grid_spacing, False),
            ("local_smoothness_weight", local_smoothness_weight, True),
            ("local_bending_weight", local_bending_weight, True),
            ("local_anchor_falloff_distance", local_anchor_falloff_distance, False),
        ):
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            valid = valid and (value >= 0 if allow_zero else value > 0)
            if not valid:
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be a {qualifier} number.")
        for name, value in (
            ("tie_maximum_displacement", tie_maximum_displacement),
            ("tie_ransac_reprojection_threshold", tie_ransac_reprojection_threshold),
            ("tie_robust_loss_scale", tie_robust_loss_scale),
        ):
            if value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive number or None.")
        JointCoregistration._parse_search_radius(tie_search_radius)
        for name, value in (("tie_orb_max_features", tie_orb_max_features), ("tie_max_matches_per_window", tie_max_matches_per_window)):
            if name == "tie_max_matches_per_window" and value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer" + (" or None." if name == "tie_max_matches_per_window" else "."))
        if moving_tie_method not in {"orb", "elastix"}:
            raise ValueError("moving_tie_method must be orb or elastix.")
        if fixed_tie_method not in {"orb", "elastix"}:
            raise ValueError("fixed_tie_method must be orb or elastix.")
        if tie_robust_loss not in {"none", "huber", "soft_l1", "cauchy"}:
            raise ValueError("tie_robust_loss must be 'none', 'huber', 'soft_l1', or 'cauchy'.")
        if resampling_method not in {"nearest", "bilinear", "cubic", "lanczos"}:
            raise ValueError("resampling_method must be 'nearest', 'bilinear', 'cubic', or 'lanczos'.")
        _validate_output_grid(tap=tap, resolution=resolution)
        if not isinstance(build_overviews, bool):
            raise ValueError("build_overviews must be a boolean.")
        for name, value in (("tie_save_path", tie_save_path), ("tie_save_crs_path", tie_save_crs_path), ("tie_load_path", tie_load_path)):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or None.")
        if tie_save_crs_path is not None and not tie_save_crs_path.strip():
            raise ValueError("tie_save_crs_path must be a non-empty vector output path or None.")
        if resume_from_outputs not in {"no", "yes", "validate"}:
            raise ValueError("resume_from_outputs must be 'no', 'yes', or 'validate'.")


