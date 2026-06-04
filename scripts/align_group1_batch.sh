#!/usr/bin/env bash
set -euo pipefail

FIXED_IMAGE="/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/intensity.tif"
INPUT_DIR="/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/group1"
OUTPUT_DIR="/home/manumea/Projects/DryForest/group1"

mkdir -p "$OUTPUT_DIR"

files=(
  "17AUG05211448-M1BS-200011893517_01_P001_cloud_masked.tif"
  "17AUG05211449-M1BS-200011893517_01_P002_cloud_masked.tif"
  "17AUG05211451-M1BS-200011893517_01_P003_cloud_masked.tif"
  "17JUL30211823-M1BS-200011908857_01_P002_cloud_masked.tif"
  "17JUL30211825-M1BS-200011908857_01_P003_cloud_masked.tif"
  "17MAY17212856-M1BS-200011964314_01_P001_cloud_masked.tif"
  "17MAY17212858-M1BS-200011964314_01_P002_cloud_masked.tif"
  "17MAY17212900-M1BS-200011964314_01_P003_cloud_masked.tif"
)

for name in "${files[@]}"; do
  in_path="$INPUT_DIR/$name"
  stem="${name%.tif}"
  out_path="$OUTPUT_DIR/${stem}_aligned.tif"

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
