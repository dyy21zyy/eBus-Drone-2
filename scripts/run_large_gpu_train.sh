#!/usr/bin/env bash
python -m src.main --mode train --config configs/default.yaml --instance large --seed 1 --method am_dueling_ddqn_dr
