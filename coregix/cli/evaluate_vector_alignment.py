#!/usr/bin/env python3
"""Evaluate vector alignment using a Coregix transform JSON."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from coregix.evaluation import evaluate_vector_alignment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate paired vector geometries after applying a Coregix "
            "source_to_target transform."
        ),
    )
    parser.add_argument("--fixed-vector", required=True, help="Reference vector file.")
    parser.add_argument("--moving-vector", required=True, help="Source vector file to transform before comparison.")
    parser.add_argument("--transform-json", required=True, help="Coregix transform JSON produced by align-image-pair.")
    parser.add_argument("--id-field", required=True, help="Field used to pair fixed and moving features.")
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=0.5,
        help="Rasterization pixel size in CRS units for boundary RMSE (default: 0.5).",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=8.0,
        help="Padding around each paired feature in CRS units (default: 8.0).",
    )
    parser.add_argument("--output-json", help="Optional JSON summary output path.")
    parser.add_argument("--output-csv", help="Optional per-feature CSV output path.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.pixel_size <= 0:
        parser.error("--pixel-size must be > 0.")
    if args.padding < 0:
        parser.error("--padding must be >= 0.")

    result = evaluate_vector_alignment(
        fixed_vector_path=args.fixed_vector,
        moving_vector_path=args.moving_vector,
        transform_json_path=args.transform_json,
        id_field=args.id_field,
        pixel_size=args.pixel_size,
        padding=args.padding,
        output_json_path=args.output_json,
        output_csv_path=args.output_csv,
    )
    print(
        json.dumps(
            {
                "rmse_m": result.rmse_m,
                "initial_rmse_m": result.initial_rmse_m,
                "aligned_rmse_m": result.aligned_rmse_m,
                "improvement_m": result.improvement_m,
                "improvement_percent": result.improvement_percent,
                "feature_count": result.feature_count,
                "output_json_path": result.output_json_path,
                "output_csv_path": result.output_csv_path,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
