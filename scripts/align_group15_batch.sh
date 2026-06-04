#!/usr/bin/env bash
set -euo pipefail

FIXED_IMAGE="/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/intensity.tif"
INPUT_DIR="/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/unaligned_groups/group15"
OUTPUT_DIR="/home/manumea/Projects/DryForest/group15"
FORCE="${FORCE:-0}"

mkdir -p "$OUTPUT_DIR"

shopt -s nullglob
files=("$INPUT_DIR"/*_cloud_masked.tif)
shopt -u nullglob

if (( ${#files[@]} == 0 )); then
  echo "No cloud-masked TIFFs found in: $INPUT_DIR" >&2
  exit 1
fi

for in_path in "${files[@]}"; do
  name="$(basename "$in_path")"
  stem="${name%.tif}"
  out_path="$OUTPUT_DIR/${stem}_aligned.tif"

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
      --solve-resolution 0.5 \
      --split-factor 1 \
      --no-output-on-moving-grid \
      --trim-edge-invalid \
      --edge-trim-depth 8 \
      --edge-trim-invalid-below -1000
done
