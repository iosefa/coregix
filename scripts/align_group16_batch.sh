#!/usr/bin/env bash
set -euo pipefail

FIXED_IMAGE="/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/intensity.tif"
INPUT_DIR="/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/group16"
OUTPUT_DIR="/home/manumea/Projects/DryForest/group16"

mkdir -p "$OUTPUT_DIR"

files=(
  "19JAN07211705-M1BS-200011964575_01_P002_cloud_masked.tif"
  "19JAN13211244-M1BS-200011893516_01_P001_cloud_masked.tif"
  "19JAN13211246-M1BS-200011893516_01_P002_cloud_masked.tif"
  "19JAN13211248-M1BS-200011893516_01_P003_cloud_masked.tif"
  "19JAN13211250-M1BS-200011893516_01_P004_cloud_masked.tif"
  "19JAN13211252-M1BS-200011893516_01_P005_cloud_masked.tif"
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
