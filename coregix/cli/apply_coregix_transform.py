#!/usr/bin/env python3
"""Apply a saved Coregix transform JSON to a raster."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from coregix.pipelines.apply_transform import apply_coregix_transform


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply a Coregix transform JSON without rerunning registration.",
    )
    parser.add_argument("--moving-image", required=True, help="Path to source raster to transform.")
    parser.add_argument("--transform-json", required=True, help="Coregix transform JSON from align-image-pair.")
    parser.add_argument("--output-image", required=True, help="Path to output transformed raster.")
    parser.add_argument(
        "--allow-moving-image-mismatch",
        action="store_true",
        help="Allow applying the transform to a raster whose path/metadata differ from the JSON.",
    )
    parser.add_argument(
        "--trim-edge-invalid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After applying, set pixels adjacent to invalid boundaries to nodata.",
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
        help="For --trim-edge-invalid, treat values <= this threshold as invalid.",
    )
    parser.add_argument(
        "--edge-trim-invalid-above",
        type=float,
        help="For --trim-edge-invalid, treat values >= this threshold as invalid.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not os.path.isfile(args.moving_image):
        parser.error(f"--moving-image does not exist: {args.moving_image}")
    if not os.path.isfile(args.transform_json):
        parser.error(f"--transform-json does not exist: {args.transform_json}")
    if args.edge_trim_depth <= 0:
        parser.error("--edge-trim-depth must be > 0.")
    if args.edge_trim_detection_band_index < 0:
        parser.error("--edge-trim-detection-band-index must be >= 0.")

    result = apply_coregix_transform(
        moving_image_path=args.moving_image,
        transform_json_path=args.transform_json,
        output_image_path=args.output_image,
        allow_moving_image_mismatch=args.allow_moving_image_mismatch,
        trim_edge_invalid=args.trim_edge_invalid,
        edge_trim_depth=args.edge_trim_depth,
        edge_trim_detection_band_index=args.edge_trim_detection_band_index,
        edge_trim_invalid_below=args.edge_trim_invalid_below,
        edge_trim_invalid_above=args.edge_trim_invalid_above,
    )
    print(
        json.dumps(
            {
                "output_image_path": result.output_image_path,
                "transform_json_path": result.transform_json_path,
                "output_width": result.output_width,
                "output_height": result.output_height,
                "output_transform": result.output_transform,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
