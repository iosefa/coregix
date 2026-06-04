#!/usr/bin/env bash
set -euo pipefail

FIXED_IMAGE="/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/intensity.tif"
INPUT_DIR="/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/unaligned_groups/group7"
OUTPUT_DIR="/home/manumea/Projects/DryForest/group7"
FORCE="${FORCE:-0}"

mkdir -p "$OUTPUT_DIR"

files=(
  "17FEB18211124-M1BS-200011758845_01_P003_cloud_masked.tif"
  "17FEB18211126-M1BS-200011758845_01_P004_cloud_masked.tif"
  "17FEB18211128-M1BS-200011758845_01_P005_cloud_masked.tif"
  "17FEB18211130-M1BS-200011758845_01_P006_cloud_masked.tif"
)

for name in "${files[@]}"; do
  in_path="$INPUT_DIR/$name"
  stem="${name%.tif}"
  out_path="$OUTPUT_DIR/${stem}_aligned.tif"

  if [[ ! -f "$in_path" ]]; then
    echo "Missing input, skipping: $in_path" >&2
    continue
  fi

  if [[ -f "$out_path" && "$FORCE" != "1" ]]; then
    echo "Skipping existing output: $out_path"
    continue
  fi

  echo "Aligning: $name"
  /usr/bin/time -f "elapsed=%E user=%U sys=%S maxrss_kb=%M" \
    python -m coregix.cli.align_image_pair \
      --moving-image "$in_path" \
      --fixed-image "$FIXED_IMAGE" \
      --output-image "$out_path" \
      --moving-band-index 4 \
      --fixed-band-index 0 \
      --solve-resolutions 8,4,0.5 \
      --split-factor 1 \
      --no-output-on-moving-grid \
      --trim-edge-invalid \
      --edge-trim-depth 8 \
      --edge-trim-invalid-below -1000
done
