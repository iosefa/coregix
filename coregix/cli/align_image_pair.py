"""Command line interface for joint raster coregistration."""

import argparse
import json
from typing import Optional

from coregix import align_image_pair


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Joint global and local raster coregistration.")
    parser.add_argument("--input-images", nargs="+", required=True, help="Moving raster paths.")
    parser.add_argument("--output-images", nargs="+", required=True, help="Output paths in input order.")
    parser.add_argument("--fixed-images", nargs="+", help="Reference rasters held fixed.")
    parser.add_argument("--options", type=json.loads, default={},
                        help="JSON object of align_image_pair keyword arguments.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not isinstance(args.options, dict):
        parser.error("--options must be a JSON object.")
    options = args.options.copy()
    if args.fixed_images is not None:
        options["fixed_images"] = args.fixed_images
    outputs = align_image_pair(args.input_images, args.output_images, **options)
    print(json.dumps({"output_images": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
