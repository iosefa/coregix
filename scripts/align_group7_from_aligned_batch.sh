#!/usr/bin/env bash
set -euo pipefail

FIXED_IMAGE="/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/intensity.tif"
INPUT_DIR="/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/group7/group7"
OUTPUT_DIR="/home/manumea/Projects/DryForest/group7"

mkdir -p "$OUTPUT_DIR"

files=(
  "17FEB18211124-M1BS-200011758845_01_P003_cloud_masked_aligned.tif"
  "17FEB18211126-M1BS-200011758845_01_P004_cloud_masked_aligned.tif"
  "17FEB18211128-M1BS-200011758845_01_P005_cloud_masked_aligned.tif"
  "17FEB18211130-M1BS-200011758845_01_P006_cloud_masked_aligned.tif"
)

for name in "${files[@]}"; do
  in_path="$INPUT_DIR/$name"
  out_path="$OUTPUT_DIR/$name"

  if [[ -f "$out_path" ]]; then
    echo "Skipping existing output: $out_path"
    continue
  fi

  echo "Aligning: $name"
  /usr/bin/time -f "elapsed=%E user=%U sys=%S maxrss_kb=%M" \
    python -m coregix.cli.align_image_pair \
      --moving-image "$in_path" \
      --fixed-image "$FIXED_IMAGE" \
      --output-image "$out_path" \
      --split-factor 2 \
      --solve-resolution 0.5 \
      --no-output-on-moving-grid \
      --trim-edge-invalid \
      --edge-trim-depth 8 \
      --edge-trim-invalid-below -1000
done
