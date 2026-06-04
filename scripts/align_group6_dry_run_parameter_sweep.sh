#!/usr/bin/env bash
set -uo pipefail

FIXED_IMAGE="${FIXED_IMAGE:-/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/intensity.tif}"
INPUT_DIR="${INPUT_DIR:-/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/unaligned_groups/group6}"
GPKG="${GPKG:-/home/manumea/Projects/DryForest/alignment_features.gpkg}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/manumea/Projects/DryForest/group6_parameter_sweep}"
FIXED_BAND_INDEX="${FIXED_BAND_INDEX:-0}"
FORCE="${FORCE:-0}"

# Tier-1 default: cheapest useful sweep for group 6.
# Examples:
#   SOLVE_RESOLUTION_SETS="6,2 6 6,1 8,2 8,4,2 4,2" bash scripts/align_group6_dry_run_parameter_sweep.sh
#   MOVING_BANDS="0 4" EDGE_MODES="edge raw" bash scripts/align_group6_dry_run_parameter_sweep.sh
#   SPLIT_FACTORS="0 1 2" MOVING_BANDS="4" SOLVE_RESOLUTION_SETS="6,2" bash scripts/align_group6_dry_run_parameter_sweep.sh
SOLVE_RESOLUTION_SETS="${SOLVE_RESOLUTION_SETS:-6,2}"
SPLIT_FACTORS="${SPLIT_FACTORS:-0}"
MOVING_BANDS="${MOVING_BANDS:-0 1 2 3 4 5 6 7}"
EDGE_MODES="${EDGE_MODES:-edge raw}"

SUMMARY_CSV="$OUTPUT_DIR/summary.csv"
VECTOR_DIR="$OUTPUT_DIR/vector_subsets"
mkdir -p "$OUTPUT_DIR" "$VECTOR_DIR"

if [[ ! -f "$FIXED_IMAGE" ]]; then
  echo "Missing fixed image: $FIXED_IMAGE" >&2
  exit 1
fi
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Missing input directory: $INPUT_DIR" >&2
  exit 1
fi
if [[ ! -f "$GPKG" ]]; then
  echo "Missing alignment vector GeoPackage: $GPKG" >&2
  exit 1
fi

printf '%s\n' 'image,stem,solve_resolutions,split_factor,moving_band_index,fixed_band_index,edge_mode,initial_rmse_m,aligned_rmse_m,rmse_m,improvement_m,improvement_percent,feature_count,transform_json,rmse_json,status' > "$SUMMARY_CSV"

shopt -s nullglob
files=("$INPUT_DIR"/*_cloud_masked.tif)
shopt -u nullglob
if (( ${#files[@]} == 0 )); then
  echo "No cloud-masked TIFFs found in: $INPUT_DIR" >&2
  exit 1
fi

read -r -a resolution_sets <<< "$SOLVE_RESOLUTION_SETS"
read -r -a split_factors <<< "$SPLIT_FACTORS"
read -r -a moving_bands <<< "$MOVING_BANDS"
read -r -a edge_modes <<< "$EDGE_MODES"

append_failure() {
  local image="$1"
  local stem="$2"
  local resolutions="$3"
  local split_factor="$4"
  local moving_band="$5"
  local edge_mode="$6"
  local transform_json="$7"
  local rmse_json="$8"
  local status="$9"
  python - "$SUMMARY_CSV" "$image" "$stem" "$resolutions" "$split_factor" "$moving_band" "$FIXED_BAND_INDEX" "$edge_mode" "$transform_json" "$rmse_json" "$status" <<'PY'
import csv
import sys
summary, image, stem, resolutions, split_factor, moving_band, fixed_band, edge_mode, transform_json, rmse_json, status = sys.argv[1:]
with open(summary, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([image, stem, resolutions, split_factor, moving_band, fixed_band, edge_mode, "", "", "", "", "", "", transform_json, rmse_json, status])
PY
}

append_success() {
  local image="$1"
  local stem="$2"
  local resolutions="$3"
  local split_factor="$4"
  local moving_band="$5"
  local edge_mode="$6"
  local transform_json="$7"
  local rmse_json="$8"
  python - "$SUMMARY_CSV" "$image" "$stem" "$resolutions" "$split_factor" "$moving_band" "$FIXED_BAND_INDEX" "$edge_mode" "$transform_json" "$rmse_json" <<'PY'
import csv
import json
import sys
summary, image, stem, resolutions, split_factor, moving_band, fixed_band, edge_mode, transform_json, rmse_json = sys.argv[1:]
with open(rmse_json, "r", encoding="utf-8") as f:
    result = json.load(f)
with open(summary, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        image,
        stem,
        resolutions,
        split_factor,
        moving_band,
        fixed_band,
        edge_mode,
        result.get("initial_rmse_m", ""),
        result.get("aligned_rmse_m", result.get("rmse_m", "")),
        result.get("rmse_m", ""),
        result.get("improvement_m", ""),
        result.get("improvement_percent", ""),
        result.get("feature_count", ""),
        transform_json,
        rmse_json,
        "ok",
    ])
PY
}

for moving_image in "${files[@]}"; do
  image_name="$(basename "$moving_image")"
  stem="${image_name%.tif}"
  fixed_vec="$VECTOR_DIR/${stem}_fixed.gpkg"
  moving_vec="$VECTOR_DIR/${stem}_moving.gpkg"

  echo "Preparing vector subsets for: $image_name"
  if ! ogr2ogr -overwrite -nln fixed_features "$fixed_vec" "$GPKG" fixed_features; then
    echo "Failed to prepare fixed vector subset for: $image_name" >&2
    continue
  fi
  if ! ogr2ogr -overwrite -nln moving_features -where "image_path = '$moving_image'" "$moving_vec" "$GPKG" moving_features; then
    echo "Failed to prepare moving vector subset for: $image_name" >&2
    continue
  fi

  for resolutions in "${resolution_sets[@]}"; do
    solve_label="${resolutions//,/_}"
    solve_label="${solve_label//./p}"
    for split_factor in "${split_factors[@]}"; do
      for moving_band_index in "${moving_bands[@]}"; do
        for edge_mode in "${edge_modes[@]}"; do
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

          run_dir="$OUTPUT_DIR/$stem/solve_${solve_label}/split_${split_factor}/band_${moving_band_index}/${edge_mode}"
          mkdir -p "$run_dir"
          transform_json="$run_dir/${stem}_transform.json"
          rmse_json="$run_dir/${stem}_rmse.json"
          rmse_csv="$run_dir/${stem}_rmse.csv"
          status_file="$run_dir/status.txt"

          if [[ -f "$rmse_json" && "$FORCE" != "1" ]]; then
            echo "Skipping existing result: $rmse_json"
            append_success "$image_name" "$stem" "$resolutions" "$split_factor" "$moving_band_index" "$edge_mode" "$transform_json" "$rmse_json"
            continue
          fi

          echo "Dry-run aligning: $image_name"
          echo "  solve-resolutions: $resolutions"
          echo "  split-factor: $split_factor"
          echo "  moving-band-index: $moving_band_index"
          echo "  fixed-band-index: $FIXED_BAND_INDEX"
          echo "  edge mode: $edge_mode"
          echo "  run dir: $run_dir"

          /usr/bin/time -f "elapsed=%E user=%U sys=%S maxrss_kb=%M" \
            python -m coregix.cli.align_image_pair \
              --moving-image "$moving_image" \
              --fixed-image "$FIXED_IMAGE" \
              --moving-band-index "$moving_band_index" \
              --fixed-band-index "$FIXED_BAND_INDEX" \
              --solve-resolutions "$resolutions" \
              --split-factor "$split_factor" \
              $edge_flag \
              --no-output-on-moving-grid \
              --trim-edge-invalid \
              --edge-trim-depth 8 \
              --edge-trim-invalid-below -1000 \
              --dry-run \
              --output-transform-json "$transform_json"
          align_status=$?
          if [[ "$align_status" -ne 0 ]]; then
            echo "align failed exit_code=$align_status" | tee "$status_file" >&2
            append_failure "$image_name" "$stem" "$resolutions" "$split_factor" "$moving_band_index" "$edge_mode" "$transform_json" "$rmse_json" "align_failed_$align_status"
            continue
          fi

          python -m coregix.cli.evaluate_vector_alignment \
            --fixed-vector "$fixed_vec" \
            --moving-vector "$moving_vec" \
            --transform-json "$transform_json" \
            --id-field feature_id \
            --output-json "$rmse_json" \
            --output-csv "$rmse_csv"
          eval_status=$?
          if [[ "$eval_status" -ne 0 ]]; then
            echo "evaluate failed exit_code=$eval_status" | tee "$status_file" >&2
            append_failure "$image_name" "$stem" "$resolutions" "$split_factor" "$moving_band_index" "$edge_mode" "$transform_json" "$rmse_json" "eval_failed_$eval_status"
            continue
          fi

          echo "ok" > "$status_file"
          append_success "$image_name" "$stem" "$resolutions" "$split_factor" "$moving_band_index" "$edge_mode" "$transform_json" "$rmse_json"
        done
      done
    done
  done
done

echo "Summary written to: $SUMMARY_CSV"
echo "Best rows by aligned RMSE:"
python - "$SUMMARY_CSV" <<'PY'
import csv
import math
import sys
summary = sys.argv[1]
with open(summary, newline="", encoding="utf-8") as f:
    rows = [row for row in csv.DictReader(f) if row.get("status") == "ok" and row.get("aligned_rmse_m")]
rows.sort(key=lambda row: float(row["aligned_rmse_m"]) if row["aligned_rmse_m"] else math.inf)
for row in rows[:10]:
    print(
        f"aligned_rmse_m={float(row['aligned_rmse_m']):.3f} "
        f"image={row['image']} solve={row['solve_resolutions']} split={row['split_factor']} "
        f"band={row['moving_band_index']} edge={row['edge_mode']} "
        f"initial={float(row['initial_rmse_m']):.3f} improvement={float(row['improvement_m']):.3f}"
    )
PY
