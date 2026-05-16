#!/usr/bin/env bash
set -euo pipefail
CONFIG=${CONFIG:-configs/default.yaml}
INSTANCE=${INSTANCE:-large}
METHOD=${METHOD:-am_dueling_ddqn_dr}
EPISODES=${EPISODES:-5000}
OUT=${OUT:-outputs_gpu_large}
SEEDS=(${SEEDS:-1 2 3})
for SEED in "${SEEDS[@]}"; do
  python -m src.main --mode generate --config "$CONFIG" --instance "$INSTANCE" --seed "$SEED" --output-dir "$OUT" --overwrite
  python -m src.main --mode offline --config "$CONFIG" --instance "$INSTANCE" --seed "$SEED" --output-dir "$OUT" --overwrite
  python -m src.main --mode train --config "$CONFIG" --instance "$INSTANCE" --seed "$SEED" --method "$METHOD" --episodes "$EPISODES" --output-dir "$OUT" --overwrite
 done
