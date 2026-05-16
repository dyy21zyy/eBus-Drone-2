#!/usr/bin/env bash
python -m src.main --mode sensitivity --config configs/default.yaml --instance large --seeds 1 --methods am_dueling_ddqn_dr --sensitivity passenger_intensity --values 0.5 1.0 1.5
