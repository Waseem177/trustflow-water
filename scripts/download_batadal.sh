#!/usr/bin/env bash
# Fetch the BATADAL dataset as used by Erba et al. (scy-phy/ICS-Evasion-Attacks),
# so the detector and the concealment attacks operate on identical data.
set -euo pipefail
BASE="https://raw.githubusercontent.com/scy-phy/ICS-Evasion-Attacks/master/Data/BATADAL"
OUT="$(dirname "$0")/../data/batadal"
mkdir -p "$OUT/attacks"
for f in train_dataset.csv test_dataset_1.csv test_dataset_2.csv; do
  echo "fetching $f"; curl -fsSL --max-time 60 -o "$OUT/$f" "$BASE/$f"
done
for i in $(seq 1 14); do
  f="attack_${i}_from_test_dataset.csv"
  echo "fetching $f"; curl -fsSL --max-time 60 -o "$OUT/attacks/$f" "$BASE/$f"
done
echo "done -> $OUT"
