#!/usr/bin/env bash
python -m src.main --mode benchmark --config configs/default.yaml --instances large --seeds 1 2 3 --methods am_dueling_ddqn_dr ddqn_dr am_ddqn_dr
