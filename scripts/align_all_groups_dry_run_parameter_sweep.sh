#!/usr/bin/env bash
set -uo pipefail

FIXED_IMAGE="${FIXED_IMAGE:-/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/intensity.tif}"
INPUT_ROOT="${INPUT_ROOT:-/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/unaligned_groups}"
GPKG="${GPKG:-/home/manumea/Projects/DryForest/alignment_features.gpkg}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/manumea/Projects/DryForest/all_groups_parameter_sweep}"
FIXED_BAND_INDEX="${FIXED_BAND_INDEX:-0}"
FORCE="${FORCE:-0}"
ACCEPT_THRESHOLD="${ACCEPT_THRESHOLD:-}"
IMAGE_LIST="${IMAGE_LIST:-}"

# Candidate order is based on the group 6 and group 7 experiments:
# - raw intensity often beat edge proxy for vector QA
# - a single coarse solve around 6 m was strong
# - bands 7/5 were strong for group 6, while 4/0 keep coverage for other groups/tests
# Format: solve_resolutions|split_factor|moving_band_index|edge_mode
# Override by setting CANDIDATES to newline-separated rows in the same format.
DEFAULT_CANDIDATES=$'6|0|7|raw\n6|0|5|raw\n6|0|4|raw\n6|0|0|raw\n4|0|7|raw\n4|0|5|raw\n4|0|4|raw\n6,2|0|7|raw\n6,2|0|5|raw\n6,2|0|4|raw\n6,2|0|0|raw\n6|0|7|edge\n6|0|5|edge\n6|0|4|edge\n6|0|0|edge\n6,2|0|7|edge\n6,2|0|5|edge\n6,2|0|4|edge\n6,2|0|0|edge'
CANDIDATES="${CANDIDATES:-$DEFAULT_CANDIDATES}"

SUMMARY_CSV="$OUTPUT_DIR/summary.csv"
BEST_CSV="$OUTPUT_DIR/best_by_image.csv"
VECTOR_DIR="$OUTPUT_DIR/vector_subsets"
mkdir -p "$OUTPUT_DIR" "$VECTOR_DIR"

if [[ ! -f "$FIXED_IMAGE" ]]; then
  echo "Missing fixed image: $FIXED_IMAGE" >&2
  exit 1
fi
if [[ ! -d "$INPUT_ROOT" ]]; then
  echo "Missing input root: $INPUT_ROOT" >&2
  exit 1
fi
if [[ -n "$IMAGE_LIST" && ! -f "$IMAGE_LIST" ]]; then
  echo "Missing image list: $IMAGE_LIST" >&2
  exit 1
fi
if [[ ! -f "$GPKG" ]]; then
  echo "Missing alignment vector GeoPackage: $GPKG" >&2
  exit 1
fi

printf '%s\n' 'image_path,image,group_name,stem,solve_resolutions,split_factor,moving_band_index,fixed_band_index,edge_mode,initial_rmse_m,aligned_rmse_m,rmse_m,improvement_m,improvement_percent,feature_count,transform_json,rmse_json,status' > "$SUMMARY_CSV"

append_failure() {
  local image_path="$1"
  local image="$2"
  local group_name="$3"
  local stem="$4"
  local resolutions="$5"
  local split_factor="$6"
  local moving_band="$7"
  local edge_mode="$8"
  local transform_json="$9"
  local rmse_json="${10}"
  local status="${11}"
  python - "$SUMMARY_CSV" "$image_path" "$image" "$group_name" "$stem" "$resolutions" "$split_factor" "$moving_band" "$FIXED_BAND_INDEX" "$edge_mode" "$transform_json" "$rmse_json" "$status" <<'PY'
import csv
import sys
summary, image_path, image, group_name, stem, resolutions, split_factor, moving_band, fixed_band, edge_mode, transform_json, rmse_json, status = sys.argv[1:]
with open(summary, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([image_path, image, group_name, stem, resolutions, split_factor, moving_band, fixed_band, edge_mode, "", "", "", "", "", "", transform_json, rmse_json, status])
PY
}

append_success() {
  local image_path="$1"
  local image="$2"
  local group_name="$3"
  local stem="$4"
  local resolutions="$5"
  local split_factor="$6"
  local moving_band="$7"
  local edge_mode="$8"
  local transform_json="$9"
  local rmse_json="${10}"
  python - "$SUMMARY_CSV" "$image_path" "$image" "$group_name" "$stem" "$resolutions" "$split_factor" "$moving_band" "$FIXED_BAND_INDEX" "$edge_mode" "$transform_json" "$rmse_json" <<'PY'
import csv
import json
import sys
summary, image_path, image, group_name, stem, resolutions, split_factor, moving_band, fixed_band, edge_mode, transform_json, rmse_json = sys.argv[1:]
with open(rmse_json, "r", encoding="utf-8") as f:
    result = json.load(f)
with open(summary, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        image_path,
        image,
        group_name,
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

write_best_csv() {
  python - "$SUMMARY_CSV" "$BEST_CSV" <<'PY'
import csv
import math
import sys
summary_csv, best_csv = sys.argv[1:]
rows = []
with open(summary_csv, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("status") == "ok" and row.get("aligned_rmse_m"):
            row["_aligned"] = float(row["aligned_rmse_m"])
            rows.append(row)
best = {}
for row in rows:
    key = row["image_path"]
    if key not in best or row["_aligned"] < best[key]["_aligned"]:
        best[key] = row
fieldnames = [
    "image_path", "image", "group_name", "stem", "solve_resolutions", "split_factor",
    "moving_band_index", "fixed_band_index", "edge_mode", "initial_rmse_m",
    "aligned_rmse_m", "rmse_m", "improvement_m", "improvement_percent",
    "feature_count", "transform_json", "rmse_json", "status",
]
with open(best_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in sorted(best.values(), key=lambda r: (r["group_name"], r["image"])):
        clean = {name: row.get(name, "") for name in fieldnames}
        writer.writerow(clean)
print(f"Best-by-image written to: {best_csv}")
print("Best rows by worst-first RMSE:")
for row in sorted(best.values(), key=lambda r: r["_aligned"], reverse=True)[:15]:
    print(
        f"aligned_rmse_m={row['_aligned']:.3f} image={row['image']} group={row['group_name']} "
        f"solve={row['solve_resolutions']} split={row['split_factor']} "
        f"band={row['moving_band_index']} edge={row['edge_mode']}"
    )
PY
}

if [[ -n "$IMAGE_LIST" ]]; then
  mapfile -t files < <(sed '/^[[:space:]]*$/d; /^[[:space:]]*#/d' "$IMAGE_LIST")
else
  mapfile -t files < <(find "$INPUT_ROOT" -type f \( -iname '*_cloud_masked.tif' -o -iname '*_cloud_masked.tiff' \) | sort)
fi
if (( ${#files[@]} == 0 )); then
  if [[ -n "$IMAGE_LIST" ]]; then
    echo "No image paths found in: $IMAGE_LIST" >&2
  else
    echo "No cloud-masked TIFFs found under: $INPUT_ROOT" >&2
  fi
  exit 1
fi
for moving_image in "${files[@]}"; do
  if [[ ! -f "$moving_image" ]]; then
    echo "Image list entry does not exist: $moving_image" >&2
    exit 1
  fi
done
candidate_count=$(printf '%s\n' "$CANDIDATES" | sed '/^[[:space:]]*$/d' | wc -l)
echo "Images: ${#files[@]}"
if [[ -n "$IMAGE_LIST" ]]; then
  echo "Image list: $IMAGE_LIST"
fi
echo "Candidates per image: $candidate_count"
echo "Output dir: $OUTPUT_DIR"
if [[ -n "$ACCEPT_THRESHOLD" ]]; then
  echo "Early stop threshold: $ACCEPT_THRESHOLD m"
else
  echo "Early stop threshold: disabled; all candidates will be tested"
fi

for moving_image in "${files[@]}"; do
  image_name="$(basename "$moving_image")"
  stem="${image_name%.*}"
  group_name="$(basename "$(dirname "$moving_image")")"
  safe_image_dir="$OUTPUT_DIR/$group_name/$stem"
  fixed_vec="$VECTOR_DIR/${group_name}_${stem}_fixed.gpkg"
  moving_vec="$VECTOR_DIR/${group_name}_${stem}_moving.gpkg"

  echo ""
  echo "Preparing vector subsets for: $group_name/$image_name"
  if ! ogr2ogr -overwrite -nln fixed_features "$fixed_vec" "$GPKG" fixed_features; then
    echo "Failed to prepare fixed vector subset for: $moving_image" >&2
    continue
  fi
  if ! ogr2ogr -overwrite -nln moving_features -where "image_path = '$moving_image'" "$moving_vec" "$GPKG" moving_features; then
    echo "Failed to prepare moving vector subset for: $moving_image" >&2
    continue
  fi
  moving_feature_count=$(ogrinfo -so "$moving_vec" moving_features 2>/dev/null | awk -F': ' '/Feature Count/ {print $2}')
  if [[ -z "$moving_feature_count" || "$moving_feature_count" == "0" ]]; then
    echo "No moving features for image, skipping: $moving_image" >&2
    continue
  fi

  best_for_image=""
  while IFS='|' read -r resolutions split_factor moving_band_index edge_mode; do
    [[ -z "${resolutions//[[:space:]]/}" ]] && continue
    resolutions="${resolutions//[[:space:]]/}"
    split_factor="${split_factor//[[:space:]]/}"
    moving_band_index="${moving_band_index//[[:space:]]/}"
    edge_mode="${edge_mode//[[:space:]]/}"
    case "$edge_mode" in
      edge) edge_flag="--use-edge-proxies" ;;
      raw) edge_flag="--no-use-edge-proxies" ;;
      *) echo "Unknown edge mode in candidate: $edge_mode" >&2; continue ;;
    esac

    solve_label="${resolutions//,/_}"
    solve_label="${solve_label//./p}"
    run_dir="$safe_image_dir/solve_${solve_label}/split_${split_factor}/band_${moving_band_index}/${edge_mode}"
    mkdir -p "$run_dir"
    transform_json="$run_dir/${stem}_transform.json"
    rmse_json="$run_dir/${stem}_rmse.json"
    rmse_csv="$run_dir/${stem}_rmse.csv"
    status_file="$run_dir/status.txt"

    if [[ -f "$rmse_json" && "$FORCE" != "1" ]]; then
      echo "Skipping existing result: $group_name/$image_name solve=$resolutions band=$moving_band_index edge=$edge_mode"
      append_success "$moving_image" "$image_name" "$group_name" "$stem" "$resolutions" "$split_factor" "$moving_band_index" "$edge_mode" "$transform_json" "$rmse_json"
    else
      echo "Dry-run aligning: $group_name/$image_name"
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
        append_failure "$moving_image" "$image_name" "$group_name" "$stem" "$resolutions" "$split_factor" "$moving_band_index" "$edge_mode" "$transform_json" "$rmse_json" "align_failed_$align_status"
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
        append_failure "$moving_image" "$image_name" "$group_name" "$stem" "$resolutions" "$split_factor" "$moving_band_index" "$edge_mode" "$transform_json" "$rmse_json" "eval_failed_$eval_status"
        continue
      fi
      echo "ok" > "$status_file"
      append_success "$moving_image" "$image_name" "$group_name" "$stem" "$resolutions" "$split_factor" "$moving_band_index" "$edge_mode" "$transform_json" "$rmse_json"
    fi

    current_rmse=$(python - "$rmse_json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    payload = json.load(f)
print(payload.get("aligned_rmse_m", payload.get("rmse_m", "")))
PY
)
    if [[ -n "$current_rmse" ]]; then
      best_for_image=$(python - "$best_for_image" "$current_rmse" <<'PY'
import math
import sys
old = sys.argv[1]
new = float(sys.argv[2])
if not old:
    print(new)
else:
    print(min(float(old), new))
PY
)
      echo "  current aligned RMSE: $current_rmse m; best for image: $best_for_image m"
    fi

    if [[ -n "$ACCEPT_THRESHOLD" && -n "$best_for_image" ]]; then
      should_stop=$(python - "$best_for_image" "$ACCEPT_THRESHOLD" <<'PY'
import sys
print("1" if float(sys.argv[1]) <= float(sys.argv[2]) else "0")
PY
)
      if [[ "$should_stop" == "1" ]]; then
        echo "  Accept threshold reached for $group_name/$image_name; moving to next image."
        break
      fi
    fi
  done <<< "$CANDIDATES"
done

write_best_csv
echo "Summary written to: $SUMMARY_CSV"
echo "Best-by-image CSV: $BEST_CSV"
