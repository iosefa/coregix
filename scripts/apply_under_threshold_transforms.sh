#!/usr/bin/env bash
set -euo pipefail

SUMMARY_CSV="${SUMMARY_CSV:-/home/manumea/Projects/DryForest/all_groups_parameter_sweep/summary.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/manumea/Projects/DryForest/aligned_under_1m}"
THRESHOLD="${THRESHOLD:-1.0}"
FORCE="${FORCE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRIM_EDGE_INVALID="${TRIM_EDGE_INVALID:-1}"
EDGE_TRIM_DEPTH="${EDGE_TRIM_DEPTH:-8}"
EDGE_TRIM_INVALID_BELOW="${EDGE_TRIM_INVALID_BELOW:--1000}"

if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "Missing summary CSV: $SUMMARY_CSV" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Summary CSV: $SUMMARY_CSV"
echo "Output dir: $OUTPUT_DIR"
echo "Threshold: $THRESHOLD m"

mapfile -t rows < <("$PYTHON_BIN" - "$SUMMARY_CSV" "$OUTPUT_DIR" "$THRESHOLD" <<'INNERPY'
import csv
import sys
from collections import defaultdict
from pathlib import Path

summary_csv, output_dir, threshold = sys.argv[1], sys.argv[2], float(sys.argv[3])
rows = []
with open(summary_csv, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("status") != "ok" or not row.get("aligned_rmse_m"):
            continue
        try:
            row["_rmse"] = float(row["aligned_rmse_m"])
        except ValueError:
            continue
        rows.append(row)

by_image = defaultdict(list)
for row in rows:
    by_image[row["image_path"]].append(row)

selected = []
for image_path, image_rows in by_image.items():
    best = min(image_rows, key=lambda row: row["_rmse"])
    if best["_rmse"] <= threshold:
        group = best.get("group_name") or Path(image_path).parent.name
        stem = Path(image_path).stem
        output_path = Path(output_dir) / group / f"{stem}_aligned.tif"
        selected.append((group, best.get("image", Path(image_path).name), best["_rmse"], image_path, best["transform_json"], str(output_path)))

for group, image, rmse, image_path, transform_json, output_path in sorted(selected):
    print("\t".join([image_path, transform_json, output_path, f"{rmse:.6f}", group, image]))
INNERPY
)

echo "Transforms to apply: ${#rows[@]}"

for row in "${rows[@]}"; do
  IFS=$'\t' read -r moving_image transform_json output_image rmse group image_name <<< "$row"
  if [[ -f "$output_image" && "$FORCE" != "1" ]]; then
    echo "Skipping existing: $group/$image_name -> $output_image"
    continue
  fi
  mkdir -p "$(dirname "$output_image")"
  echo "Applying: $group/$image_name rmse=${rmse}m"
  echo "  output: $output_image"
  args=(
    -m coregix.cli.apply_coregix_transform
    --moving-image "$moving_image"
    --transform-json "$transform_json"
    --output-image "$output_image"
  )
  if [[ "$TRIM_EDGE_INVALID" == "1" ]]; then
    args+=(
      --trim-edge-invalid
      --edge-trim-depth "$EDGE_TRIM_DEPTH"
      --edge-trim-invalid-below "$EDGE_TRIM_INVALID_BELOW"
    )
  fi
  "$PYTHON_BIN" "${args[@]}"
done
