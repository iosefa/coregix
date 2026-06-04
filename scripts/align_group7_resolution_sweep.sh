#!/usr/bin/env bash
set -uo pipefail

FIXED_IMAGE="${FIXED_IMAGE:-/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/intensity.tif}"
INPUT_DIR="${INPUT_DIR:-/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/unaligned_groups/group7}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/manumea/Projects/DryForest/group7_resolution_sweep}"
FORCE="${FORCE:-0}"

resolution_sets=(
  "6"
)

split_factors=(
  "0"
)

files=(
  "17FEB18211126-M1BS-200011758845_01_P004_cloud_masked.tif"
)

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$FIXED_IMAGE" ]]; then
  echo "Missing fixed image: $FIXED_IMAGE" >&2
  exit 1
fi

for split_factor in "${split_factors[@]}"; do
  for resolutions in "${resolution_sets[@]}"; do
    if [[ "$split_factor" == "1" && "$resolutions" == *"0.5"* ]]; then
      echo "Skipping split-factor 1 for solve-resolutions $resolutions; 0.5m solve is too memory-heavy with 2 chunks."
      continue
    fi
    label="${resolutions//,/_}"
    label="${label//./p}"
    set_output_dir="$OUTPUT_DIR/split_${split_factor}/solve_${label}"
    mkdir -p "$set_output_dir"

    for name in "${files[@]}"; do
      in_path="$INPUT_DIR/$name"
      stem="${name%.tif}"
      out_path="$set_output_dir/${stem}_band_0_split_${split_factor}_solve_${label}_aligned.tif"

      if [[ ! -f "$in_path" ]]; then
        echo "Missing input, skipping: $in_path" >&2
        continue
      fi

      if [[ -f "$out_path" && "$FORCE" != "1" ]]; then
        echo "Skipping existing output: $out_path"
        continue
      fi

      echo "Aligning: $name"
      echo "  split-factor: $split_factor"
      echo "  solve-resolutions: $resolutions"
      echo "  output: $out_path"
      /usr/bin/time -f "elapsed=%E user=%U sys=%S maxrss_kb=%M" \
        python -m coregix.cli.align_image_pair \
          --moving-image "$in_path" \
          --fixed-image "$FIXED_IMAGE" \
          --output-image "$out_path" \
          --moving-band-index 0 \
          --fixed-band-index 0 \
          --solve-resolutions "$resolutions" \
          --split-factor "$split_factor" \
          --no-output-on-moving-grid \
          --trim-edge-invalid \
          --edge-trim-depth 8 \
          --edge-trim-invalid-below -1000
      run_status=$?
      if [[ "$run_status" -ne 0 ]]; then
        echo "Run failed with exit code $run_status; continuing sweep." >&2
        echo "failed exit_code=$run_status" > "${out_path}.status"
        continue
      fi
      echo "ok" > "${out_path}.status"
    done
  done
done
