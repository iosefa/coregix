#!/usr/bin/env bash
set -euo pipefail

FIXED_IMAGE="/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/intensity.tif"
INPUT_DIR="/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/group10"
OUTPUT_DIR="/home/manumea/Projects/DryForest/group10"

mkdir -p "$OUTPUT_DIR"

files=(
  "18JAN02211804-M1BS-200009592532_01_P002_cloud_masked.tif"
  "18JAN02211806-M1BS-200009592532_01_P001_cloud_masked.tif"
  "18JAN02211821-M1BS-200011893504_01_P001_cloud_masked.tif"
  "18JAN02211823-M1BS-200011893504_01_P002_cloud_masked.tif"
  "18JAN02211825-M1BS-200011893504_01_P003_cloud_masked.tif"
  "18JAN02211827-M1BS-200011893504_01_P004_cloud_masked.tif"
  "18JAN02211829-M1BS-200011893504_01_P005_cloud_masked.tif"
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
      --moving-band-index 4 \
      --fixed-band-index 0 \
      --solve-resolution 0.5 \
      --split-factor 1 \
      --no-output-on-moving-grid \
      --trim-edge-invalid \
      --edge-trim-depth 8 \
      --edge-trim-invalid-below -1000
done
