#!/usr/bin/env python3
"""Trim pixels adjacent to invalid regions from an aligned raster."""

import argparse
import json
import os
import sys
from typing import Optional

from coregix.postprocess import trim_edge_invalid_pixels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Trim a few pixels around irregular invalid boundaries in an "
            "already-aligned raster."
        )
    )
    parser.add_argument("--input-image", required=True, help="Path to input aligned raster.")
    parser.add_argument(
        "--output-image",
        help="Optional output raster path. Omit only when using --in-place.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Modify the input raster in place instead of writing a copy.",
    )
    parser.add_argument(
        "--edge-depth",
        type=int,
        default=8,
        help="Number of valid pixels to trim around each invalid boundary (default: 8).",
    )
    parser.add_argument(
        "--detection-band-index",
        type=int,
        default=0,
        help="0-based band index used to detect edge artifacts (default: 0).",
    )
    parser.add_argument(
        "--invalid-below",
        type=float,
        help=(
            "Treat values <= this threshold as invalid when scanning the boundary. "
            "Use this for interpolation artifacts that are not exact nodata."
        ),
    )
    parser.add_argument(
        "--invalid-above",
        type=float,
        help="Optional upper threshold; values >= this are treated as invalid.",
    )
    parser.add_argument(
        "--nodata-value",
        type=float,
        help="Optional nodata override. Defaults to the raster's declared nodata.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not os.path.isfile(args.input_image):
        parser.error(f"--input-image does not exist: {args.input_image}")
    if args.edge_depth <= 0:
        parser.error("--edge-depth must be > 0.")
    if args.detection_band_index < 0:
        parser.error("--detection-band-index must be >= 0.")
    if not args.in_place and not args.output_image:
        parser.error("Provide --output-image or use --in-place.")
    if args.in_place and args.output_image:
        parser.error("Use either --output-image or --in-place, not both.")

    result = trim_edge_invalid_pixels(
        input_image_path=args.input_image,
        output_image_path=args.output_image,
        in_place=args.in_place,
        edge_depth=args.edge_depth,
        detection_band_index=args.detection_band_index,
        invalid_below=args.invalid_below,
        invalid_above=args.invalid_above,
        nodata_value=args.nodata_value,
    )

    print(
        json.dumps(
            {
                "output_image_path": result.output_image_path,
                "nodata_value": result.nodata_value,
                "pixels_trimmed": result.pixels_trimmed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
