#!/usr/bin/env bash
set -euo pipefail

FIXED_IMAGE="/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/intensity.tif"
MOVING_IMAGE="/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/group7_mosaic.tif"
OUTPUT_IMAGE="/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/group7_mosaic_solve_8_4_0p5.tif"
FORCE="${FORCE:-0}"

if [[ ! -f "$MOVING_IMAGE" ]]; then
  echo "Missing moving image: $MOVING_IMAGE" >&2
  exit 1
fi

if [[ ! -f "$FIXED_IMAGE" ]]; then
  echo "Missing fixed image: $FIXED_IMAGE" >&2
  exit 1
fi

if [[ -f "$OUTPUT_IMAGE" && "$FORCE" != "1" ]]; then
  echo "Skipping existing output: $OUTPUT_IMAGE"
  echo "Set FORCE=1 to overwrite."
  exit 0
fi

echo "Aligning group 7 mosaic"
/usr/bin/time -f "elapsed=%E user=%U sys=%S maxrss_kb=%M" \
  python -m coregix.cli.align_image_pair \
    --moving-image "$MOVING_IMAGE" \
    --fixed-image "$FIXED_IMAGE" \
    --output-image "$OUTPUT_IMAGE" \
    --moving-band-index 4 \
    --fixed-band-index 0 \
    --solve-resolutions 8,4,0.5 \
    --split-factor 4 \
    --no-output-on-moving-grid \
    --trim-edge-invalid \
    --edge-trim-depth 8 \
    --edge-trim-invalid-below -1000
