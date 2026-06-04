#!/usr/bin/env bash
set -euo pipefail

PRIMARY_SUMMARY_CSV="${PRIMARY_SUMMARY_CSV:-/home/manumea/Projects/DryForest/all_groups_parameter_sweep/summary.csv}"
EXTRA_SUMMARY_CSV="${EXTRA_SUMMARY_CSV:-/home/manumea/Projects/DryForest/close_failures_second_pass/summary.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/aligned_over_1m_best}"
THRESHOLD="${THRESHOLD:-1.0}"
FORCE="${FORCE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRIM_EDGE_INVALID="${TRIM_EDGE_INVALID:-1}"
EDGE_TRIM_DEPTH="${EDGE_TRIM_DEPTH:-8}"
EDGE_TRIM_INVALID_BELOW="${EDGE_TRIM_INVALID_BELOW:--1000}"

if [[ ! -f "$PRIMARY_SUMMARY_CSV" ]]; then
  echo "Missing primary summary CSV: $PRIMARY_SUMMARY_CSV" >&2
  exit 1
fi
if [[ -n "$EXTRA_SUMMARY_CSV" && ! -f "$EXTRA_SUMMARY_CSV" ]]; then
  echo "Missing extra summary CSV: $EXTRA_SUMMARY_CSV" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Primary summary CSV: $PRIMARY_SUMMARY_CSV"
echo "Extra summary CSV: $EXTRA_SUMMARY_CSV"
echo "Output dir: $OUTPUT_DIR"
echo "Threshold: > $THRESHOLD m"

mapfile -t rows < <("$PYTHON_BIN" - "$PRIMARY_SUMMARY_CSV" "$EXTRA_SUMMARY_CSV" "$OUTPUT_DIR" "$THRESHOLD" <<'INNERPY'
import csv
import sys
from collections import defaultdict
from pathlib import Path

primary_summary, extra_summary, output_dir, threshold = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
summary_paths = [Path(primary_summary)]
if extra_summary:
    summary_paths.append(Path(extra_summary))
rows = []
for summary_path in summary_paths:
    if not summary_path.exists():
        continue
    with summary_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok" or not row.get("aligned_rmse_m"):
                continue
            try:
                row["_rmse"] = float(row["aligned_rmse_m"])
            except ValueError:
                continue
            row["_summary"] = str(summary_path)
            rows.append(row)

by_image = defaultdict(list)
for row in rows:
    by_image[row["image_path"]].append(row)

selected = []
for image_path, image_rows in by_image.items():
    best = min(image_rows, key=lambda row: row["_rmse"])
    if best["_rmse"] > threshold:
        group = best.get("group_name") or Path(image_path).parent.name
        stem = Path(image_path).stem
        output_path = Path(output_dir) / group / f"{stem}_aligned_best_rmse_{best['_rmse']:.3f}m.tif"
        selected.append((group, best.get("image", Path(image_path).name), best["_rmse"], image_path, best["transform_json"], str(output_path), best.get("solve_resolutions", ""), best.get("moving_band_index", ""), best.get("edge_mode", "")))

for group, image, rmse, image_path, transform_json, output_path, solve, band, edge in sorted(selected):
    print("\t".join([image_path, transform_json, output_path, f"{rmse:.6f}", group, image, solve, band, edge]))
INNERPY
)

echo "Transforms to apply: ${#rows[@]}"

for row in "${rows[@]}"; do
  IFS=$'\t' read -r moving_image transform_json output_image rmse group image_name solve band edge <<< "$row"
  if [[ -f "$output_image" && "$FORCE" != "1" ]]; then
    echo "Skipping existing: $group/$image_name -> $output_image"
    continue
  fi
  mkdir -p "$(dirname "$output_image")"
  echo "Applying: $group/$image_name rmse=${rmse}m solve=${solve} band=${band} edge=${edge}"
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
