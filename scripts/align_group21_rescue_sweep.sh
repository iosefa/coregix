#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/home/manumea/Projects/DryForest/group21_rescue_sweep}"
ACCEPT_THRESHOLD="${ACCEPT_THRESHOLD:-1.0}"
FORCE="${FORCE:-0}"

IMAGE_LIST_FILE="${IMAGE_LIST_FILE:-/tmp/coregix_group21_rescue_images.txt}"
CANDIDATE_FILE="${CANDIDATE_FILE:-/tmp/coregix_group21_rescue_candidates.txt}"

cat > "$IMAGE_LIST_FILE" <<'EOF'
/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/unaligned_groups/group21/19MAY06211102-M1BS-200011908634_01_P001_cloud_masked.tif
/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/unaligned_groups/group21/19MAY06211104-M1BS-200011908634_01_P002_cloud_masked.tif
/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/unaligned_groups/group21/19MAY06211106-M1BS-200011908634_01_P003_cloud_masked.tif
/mnt/x/PROJECTS_2/Big_Island/ChangeHI_Trees/Dry_Forest/Data/Raster/unaligned_groups/group21/19OCT24212749-M1BS-200011893493_01_P004_cloud_masked.tif
EOF

cat > "$CANDIDATE_FILE" <<'EOF'
# Big-offset rescue for 19MAY P001/P002. These were still 23 m and 8 m off,
# so start coarser than the previous 4/6 m solves and test all useful bands.
16|0|7|raw
16|0|5|raw
16|0|4|raw
16|0|0|raw
16|0|7|edge
16|0|5|edge
16|0|4|edge
16|0|0|edge
12|0|7|raw
12|0|5|raw
12|0|4|raw
12|0|0|raw
12|0|7|edge
12|0|5|edge
12|0|4|edge
12|0|0|edge
10|0|7|raw
10|0|5|raw
10|0|4|raw
10|0|0|raw
10|0|7|edge
10|0|5|edge
10|0|4|edge
10|0|0|edge

# Missing-band rescue. P001/P002 did not get the full missing-band pass,
# and P004's best raw split solve suggests band sensitivity is real.
6|0|6|raw
6|0|3|raw
6|0|2|raw
6|0|1|raw
6|0|6|edge
6|0|3|edge
6|0|2|edge
6|0|1|edge
4|0|6|raw
4|0|3|raw
4|0|2|raw
4|0|1|raw
4|0|6|edge
4|0|3|edge
4|0|2|edge
4|0|1|edge

# Coarse-to-medium alternatives not covered by the first two sweeps.
16,8|0|7|raw
16,8|0|5|raw
16,8|0|4|raw
16,8|0|0|raw
16,8|0|7|edge
16,8|0|5|edge
16,8|0|4|edge
16,8|0|0|edge
16,8,4|0|7|raw
16,8,4|0|5|raw
16,8,4|0|4|raw
16,8,4|0|0|raw
12,6|0|7|raw
12,6|0|5|raw
12,6|0|4|raw
12,6|0|0|raw
12,6|0|7|edge
12,6|0|5|edge
12,6|0|4|edge
12,6|0|0|edge
10,5|0|7|raw
10,5|0|5|raw
10,5|0|4|raw
10,5|0|0|raw
8,4,2|0|7|raw
8,4,2|0|5|raw
8,4,2|0|4|raw
8,4,2|0|0|raw
8,4,2|0|7|edge
8,4,2|0|5|edge
8,4,2|0|4|edge
8,4,2|0|0|edge

# Near-threshold rescue for P003/P004. Split 2 helped P004 slightly, but
# the earlier split pass mostly used raw; try edge and finer final stages.
6,2|2|7|edge
6,2|2|5|edge
6,2|2|4|edge
6,2|2|0|edge
6,2|3|7|raw
6,2|3|5|raw
6,2|3|4|raw
6,2|3|0|raw
6,2|3|7|edge
6,2|3|5|edge
6,2|3|4|edge
6,2|3|0|edge
6,2,1|0|7|raw
6,2,1|0|5|raw
6,2,1|0|4|raw
6,2,1|0|0|raw
6,2,1|0|7|edge
6,2,1|0|5|edge
6,2,1|0|4|edge
6,2,1|0|0|edge
4,2,1|0|7|raw
4,2,1|0|5|raw
4,2,1|0|4|raw
4,2,1|0|0|raw
4,2,1|0|7|edge
4,2,1|0|5|edge
4,2,1|0|4|edge
4,2,1|0|0|edge
EOF

echo "Image list: $IMAGE_LIST_FILE"
echo "Candidate file: $CANDIDATE_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "Accept threshold: $ACCEPT_THRESHOLD m"

IMAGE_LIST="$IMAGE_LIST_FILE" \
OUTPUT_DIR="$OUTPUT_DIR" \
ACCEPT_THRESHOLD="$ACCEPT_THRESHOLD" \
FORCE="$FORCE" \
CANDIDATES="$(sed '/^[[:space:]]*#/d; /^[[:space:]]*$/d' "$CANDIDATE_FILE")" \
bash scripts/align_all_groups_dry_run_parameter_sweep.sh
