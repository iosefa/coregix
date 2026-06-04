#!/usr/bin/env python3
"""Coregister a source raster to a reference raster."""

import argparse
import json
import os
import sys
from typing import Optional

from coregix.pipelines.alignment import align_image_pair


def _parse_solve_resolutions(value: str) -> list[Optional[float]]:
    resolutions: list[Optional[float]] = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            raise argparse.ArgumentTypeError("--solve-resolutions entries must not be empty.")
        resolution = float(text)
        if resolution < 0:
            raise argparse.ArgumentTypeError("--solve-resolutions entries must be >= 0.")
        resolutions.append(None if resolution == 0 else resolution)
    if not resolutions:
        raise argparse.ArgumentTypeError("--solve-resolutions must contain at least one entry.")
    return resolutions


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for pairwise raster coregistration."""
    parser = argparse.ArgumentParser(
        description=(
            "Coregister a source raster to a reference raster. "
            "Uses edge-proxy registration by default."
        ),
    )
    parser.add_argument("--moving-image", required=True, help="Path to source raster that will be transformed.")
    parser.add_argument("--fixed-image", required=True, help="Path to reference raster used for alignment.")
    parser.add_argument("--output-image", help="Path to output coregistered raster. Required unless --dry-run is used.")
    parser.add_argument(
        "--output-transform-json",
        help=(
            "Optional path for a JSON sidecar describing the final coordinate transform. "
            "The JSON includes target_to_source and source_to_target CRS-coordinate matrices."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Estimate the transform and write --output-transform-json without writing "
            "an aligned output raster. Requires --output-transform-json."
        ),
    )
    parser.add_argument(
        "--band-index",
        type=int,
        default=0,
        help="0-based band index used for registration metric (default: 0).",
    )
    parser.add_argument(
        "--moving-band-index",
        type=int,
        help="Optional 0-based source-raster band index for registration metric.",
    )
    parser.add_argument(
        "--fixed-band-index",
        type=int,
        help="Optional 0-based reference-raster band index for registration metric.",
    )
    parser.add_argument(
        "--use-edge-proxies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use edge-proxy images rather than raw intensities for registration."
        ),
    )
    parser.add_argument(
        "--transform-model",
        choices=("rigid", "bspline"),
        default="rigid",
        help=(
            "Registration transform model. 'rigid' is the default global "
            "translation/rotation model. 'bspline' is experimental nonrigid "
            "alignment and supports only single-pass raster output."
        ),
    )
    parser.add_argument(
        "--split-factor",
        type=int,
        default=0,
        help=(
            "Split the solve and apply domains into 2^k chunks. "
            "0=no split, 1=halves, 2=quadrants, 3=octants (default: 0)."
        ),
    )
    parser.add_argument(
        "--moving-nodata",
        type=float,
        help="Optional override nodata value for source raster masking.",
    )
    parser.add_argument(
        "--fixed-nodata",
        type=float,
        help="Optional override nodata value for reference raster masking.",
    )
    parser.add_argument(
        "--output-nodata",
        type=float,
        help="Optional output nodata value. Defaults to source nodata, then reference nodata, else 0.",
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.01,
        help="Minimum valid-mask fraction required to run registration (default: 0.01).",
    )
    parser.add_argument(
        "--solve-resolution",
        type=float,
        help=(
            "Deprecated single-pass target pixel size, in raster CRS units, for the "
            "registration solve. Use --solve-resolutions with one or more values instead."
        ),
    )
    parser.add_argument(
        "--solve-resolutions",
        type=_parse_solve_resolutions,
        help=(
            "Comma-separated coarse-to-fine solve pixel sizes, in raster CRS units. "
            "Use 0 for native/reference resolution, for example 8,4,0.5 or 8,4,0."
        ),
    )
    parser.add_argument("--temp-dir", help="Optional parent directory for temporary working files.")
    parser.add_argument(
        "--keep-temp-dir",
        action="store_true",
        help="Keep the temporary working directory for debugging.",
    )
    parser.add_argument(
        "--log-to-console",
        action="store_true",
        help="Enable verbose registration backend logging.",
    )
    parser.add_argument(
        "--clip-fixed-to-moving",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip reference domain to source-raster bounds before alignment.",
    )
    parser.add_argument(
        "--output-on-moving-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write coregistered output on the source-raster grid. Disable with "
            "--no-output-on-moving-grid to write on the reference-raster grid."
        ),
    )
    parser.add_argument(
        "--trim-edge-invalid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After alignment, set pixels adjacent to irregular invalid "
            "boundaries to nodata."
        ),
    )
    parser.add_argument(
        "--edge-trim-depth",
        type=int,
        default=8,
        help="Number of pixels to trim around each invalid boundary (default: 8).",
    )
    parser.add_argument(
        "--edge-trim-detection-band-index",
        type=int,
        default=0,
        help="0-based band index used to detect edge artifacts for --trim-edge-invalid (default: 0).",
    )
    parser.add_argument(
        "--edge-trim-invalid-below",
        type=float,
        help=(
            "For --trim-edge-invalid, treat values <= this threshold as invalid. "
            "Useful for interpolation artifacts that are not exact nodata."
        ),
    )
    parser.add_argument(
        "--edge-trim-invalid-above",
        type=float,
        help="For --trim-edge-invalid, treat values >= this threshold as invalid.",
    )
    parser.add_argument(
        "--enforce-mutual-valid-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use only pixels valid in both source and reference rasters for both registration masks "
            "during alignment."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint for pairwise raster coregistration."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not os.path.isfile(args.moving_image):
        parser.error(f"--moving-image does not exist: {args.moving_image}")
    if not os.path.isfile(args.fixed_image):
        parser.error(f"--fixed-image does not exist: {args.fixed_image}")
    if args.dry_run and args.output_transform_json is None:
        parser.error("--dry-run requires --output-transform-json.")
    if not args.dry_run and args.output_image is None:
        parser.error("--output-image is required unless --dry-run is used.")
    if args.band_index < 0:
        parser.error("--band-index must be >= 0.")
    if args.moving_band_index is not None and args.moving_band_index < 0:
        parser.error("--moving-band-index must be >= 0.")
    if args.fixed_band_index is not None and args.fixed_band_index < 0:
        parser.error("--fixed-band-index must be >= 0.")
    if args.min_valid_fraction <= 0 or args.min_valid_fraction > 1:
        parser.error("--min-valid-fraction must be in (0, 1].")
    if args.solve_resolution is not None and args.solve_resolutions is not None:
        parser.error("Provide only one of --solve-resolution or --solve-resolutions.")
    if args.solve_resolution is not None and args.solve_resolution <= 0:
        parser.error("--solve-resolution must be > 0.")
    if args.split_factor < 0:
        parser.error("--split-factor must be >= 0.")
    if args.edge_trim_depth <= 0:
        parser.error("--edge-trim-depth must be > 0.")
    if args.edge_trim_detection_band_index < 0:
        parser.error("--edge-trim-detection-band-index must be >= 0.")
    if args.transform_model == "bspline":
        if args.dry_run:
            parser.error("--transform-model bspline cannot be used with --dry-run.")
        if args.output_transform_json is not None:
            parser.error("--transform-model bspline cannot write --output-transform-json.")
        if args.split_factor != 0:
            parser.error("--transform-model bspline requires --split-factor 0.")
        if args.solve_resolutions is not None and len(args.solve_resolutions) > 1:
            parser.error("--transform-model bspline supports only one solve resolution.")

    result = align_image_pair(
        moving_image_path=args.moving_image,
        fixed_image_path=args.fixed_image,
        output_image_path=args.output_image,
        band_index=args.band_index,
        moving_band_index=args.moving_band_index,
        fixed_band_index=args.fixed_band_index,
        moving_nodata=args.moving_nodata,
        fixed_nodata=args.fixed_nodata,
        output_nodata=args.output_nodata,
        min_valid_fraction=args.min_valid_fraction,
        temp_dir=args.temp_dir,
        keep_temp_dir=args.keep_temp_dir,
        log_to_console=args.log_to_console,
        clip_fixed_to_moving=args.clip_fixed_to_moving,
        output_on_moving_grid=args.output_on_moving_grid,
        trim_edge_invalid=args.trim_edge_invalid,
        edge_trim_depth=args.edge_trim_depth,
        edge_trim_detection_band_index=args.edge_trim_detection_band_index,
        edge_trim_invalid_below=args.edge_trim_invalid_below,
        edge_trim_invalid_above=args.edge_trim_invalid_above,
        enforce_mutual_valid_mask=args.enforce_mutual_valid_mask,
        use_edge_proxies=args.use_edge_proxies,
        split_factor=args.split_factor,
        solve_resolution=args.solve_resolution,
        solve_resolutions=args.solve_resolutions,
        transform_model=args.transform_model,
        output_transform_json_path=args.output_transform_json,
        dry_run=args.dry_run,
    )

    print(
        json.dumps(
            {
                "output_image_path": result.output_image_path,
                "temp_dir": result.temp_dir,
                "output_transform_json_path": result.output_transform_json_path,
                "dry_run": result.dry_run,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
