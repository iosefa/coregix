#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${OUTDIR:-/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/aligned_over_1m_best}"
FAIL_LOG="${FAIL_LOG:-/tmp/aligned_over_1m_best_overview_failures.txt}"
WAIT_FOR_APPLY="${WAIT_FOR_APPLY:-1}"

: > "$FAIL_LOG"

if [[ "$WAIT_FOR_APPLY" == "1" ]]; then
  echo "$(date -Is) waiting for apply processes"
  while pgrep -f "[a]pply_best_over_threshold_transforms.sh|[c]oregix.cli.apply_coregix_transform" >/dev/null; do
    echo "$(date -Is) still applying transforms"
    sleep 60
  done
fi

echo "$(date -Is) building external sidecar overviews in: $OUTDIR"
if [[ ! -d "$OUTDIR" ]]; then
  echo "Missing output dir: $OUTDIR" >&2
  exit 1
fi

count=0
skipped=0
failed=0
while IFS= read -r tif; do
  if [[ -f "$tif.ovr" ]]; then
    skipped=$((skipped + 1))
    echo "skipping existing sidecar: $tif.ovr"
    continue
  fi

  count=$((count + 1))
  echo "[$count] building sidecar overviews: $tif"
  if gdaladdo -ro \
    --config COMPRESS_OVERVIEW DEFLATE \
    --config PREDICTOR_OVERVIEW 2 \
    --config BIGTIFF_OVERVIEW IF_SAFER \
    "$tif" 2 4 8 16 32 64; then
    :
  else
    failed=$((failed + 1))
    echo "$tif" >> "$FAIL_LOG"
    rm -f "$tif.ovr"
    echo "failed, removed partial sidecar: $tif"
  fi
done < <(find "$OUTDIR" -type f -name "*.tif" | sort)

echo "$(date -Is) done attempted=$count skipped_existing=$skipped failed=$failed"
if [[ "$failed" -gt 0 ]]; then
  echo "failure log: $FAIL_LOG"
  cat "$FAIL_LOG"
fi
