#!/usr/bin/env bash
set -euo pipefail

FIXED_IMAGE="/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/intensity.tif"
INPUT_DIR="/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/group4"
OUTPUT_DIR="/home/manumea/Projects/DryForest/group4"

mkdir -p "$OUTPUT_DIR"

files=(
  "17DEC08211757-M1BS-016445319010_01_P002_cloud_masked.tif"
  "17DEC08211758-M1BS-016445319010_01_P003_cloud_masked.tif"
  "17DEC08211800-M1BS-016445319010_01_P004_cloud_masked.tif"
  "17DEC08211801-M1BS-016445319010_01_P005_cloud_masked.tif"
  "17DEC08211840-M1BS-016445318010_01_P015_cloud_masked.tif"
  "17DEC08211841-M1BS-016445318010_01_P016_cloud_masked.tif"
  "18JAN15210358-M1BS-200011893433_01_P001_cloud_masked.tif"
)

for name in "${files[@]}"; do
  in_path="$INPUT_DIR/$name"
  stem="${name%.tif}"
  out_path="$OUTPUT_DIR/${stem}_aligned.tif"

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
