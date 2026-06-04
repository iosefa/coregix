#!/usr/bin/env bash
set -uo pipefail

FIXED_IMAGE="${FIXED_IMAGE:-/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/intensity.tif}"
MOVING_IMAGE="${MOVING_IMAGE:-/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/unaligned_groups/group7/17FEB18211126-M1BS-200011758845_01_P004_cloud_masked.tif}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/manumea/Projects/DryForest/group7_parameter_sweep}"
SOLVE_RESOLUTIONS="${SOLVE_RESOLUTIONS:-8,4,1}"
SPLIT_FACTOR="${SPLIT_FACTOR:-2}"
FIXED_BAND_INDEX="${FIXED_BAND_INDEX:-0}"
FORCE="${FORCE:-0}"

# Default is a cheap representative band sweep. Override with, for example:
#   MOVING_BANDS="0 1 2 3 4 5 6 7" bash scripts/align_group7_parameter_sweep.sh
MOVING_BANDS=(${MOVING_BANDS:-0 2 4 6})
EDGE_MODES=(${EDGE_MODES:-edge raw})

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$FIXED_IMAGE" ]]; then
  echo "Missing fixed image: $FIXED_IMAGE" >&2
  exit 1
fi

if [[ ! -f "$MOVING_IMAGE" ]]; then
  echo "Missing moving image: $MOVING_IMAGE" >&2
  exit 1
fi

moving_name="$(basename "$MOVING_IMAGE")"
stem="${moving_name%.tif}"
solve_label="${SOLVE_RESOLUTIONS//,/_}"
solve_label="${solve_label//./p}"

for moving_band_index in "${MOVING_BANDS[@]}"; do
  for edge_mode in "${EDGE_MODES[@]}"; do
    case "$edge_mode" in
      edge)
        edge_flag="--use-edge-proxies"
        ;;
      raw)
        edge_flag="--no-use-edge-proxies"
        ;;
      *)
        echo "Unknown edge mode: $edge_mode. Use edge or raw." >&2
        continue
        ;;
    esac

    set_output_dir="$OUTPUT_DIR/split_${SPLIT_FACTOR}/solve_${solve_label}/band_${moving_band_index}/${edge_mode}"
    mkdir -p "$set_output_dir"
    out_path="$set_output_dir/${stem}_band_${moving_band_index}_${edge_mode}_split_${SPLIT_FACTOR}_solve_${solve_label}_aligned.tif"

    if [[ -f "$out_path" && "$FORCE" != "1" ]]; then
      echo "Skipping existing output: $out_path"
      continue
    fi

    echo "Aligning: $moving_name"
    echo "  moving-band-index: $moving_band_index"
    echo "  fixed-band-index: $FIXED_BAND_INDEX"
    echo "  edge mode: $edge_mode"
    echo "  split-factor: $SPLIT_FACTOR"
    echo "  solve-resolutions: $SOLVE_RESOLUTIONS"
    echo "  output: $out_path"

    /usr/bin/time -f "elapsed=%E user=%U sys=%S maxrss_kb=%M" \
      python -m coregix.cli.align_image_pair \
        --moving-image "$MOVING_IMAGE" \
        --fixed-image "$FIXED_IMAGE" \
        --output-image "$out_path" \
        --moving-band-index "$moving_band_index" \
        --fixed-band-index "$FIXED_BAND_INDEX" \
        --solve-resolutions "$SOLVE_RESOLUTIONS" \
        --split-factor "$SPLIT_FACTOR" \
        $edge_flag \
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
