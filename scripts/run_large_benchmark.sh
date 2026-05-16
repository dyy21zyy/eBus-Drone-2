#!/usr/bin/env bash
set -euo pipefail
CONFIG=${CONFIG:-configs/default.yaml}
OUT=${OUT:-outputs_gpu_large}
python -m src.main --mode benchmark --config "$CONFIG" --instances large --seeds 1 2 3 --methods am_dueling_ddqn_dr ddqn_dr am_ddqn_dr --output-dir "$OUT"
