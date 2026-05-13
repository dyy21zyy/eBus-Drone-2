# ebus_drone_rl

## Purpose
This repository is a reproducible research scaffold for joint **electric bus charging/control** and **drone-assisted parcel delivery** experiments under uncertainty.

## Relationship to the paper model
The code mirrors the paper workflow:
1. Generate stochastic network + demand instances.
2. Solve offline assignment (MILP-style planning constraints).
3. Train/evaluate RL and heuristic dispatch policies.
4. Aggregate benchmark/ablation/scalability/sensitivity outputs.
5. Export publication tables.

## Environment setup
- Python: 3.10+
- OS: Linux/macOS recommended

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Install dependencies
Alternative editable install:
```bash
pip install -e .
```

## CPU/GPU recommendations
- **Offline assignment** is CPU/MILP-oriented and does not require GPU.
- **RL training** can leverage GPU (PyTorch), especially for medium/large experiments.
- Suggested: 8+ CPU cores, 16GB+ RAM; GPU optional for training speed.

## Data + experiment workflow
### 1) Generate instances
```bash
python -m src.main --mode generate --config configs/default.yaml --instance small --seed 1
```

### 2) Solve offline assignment
```bash
python -m src.main --mode offline --config configs/default.yaml --instance small --seed 1
```

### 3) Train RL methods
```bash
python -m src.main --mode train --config configs/default.yaml --instance small --method proposed --seed 1
```

### 4) Evaluate policies
```bash
python -m src.main --mode eval --config configs/default.yaml --instance small --method uniform_30 --seed 1
```

## Experiment tiers
### Quick smoke test (fast sanity)
```bash
python -m src.main --mode pipeline --config configs/default.yaml --instance small --seeds 1 --methods uniform_30 battery_threshold --smoke
```

### Small debug run
```bash
python -m src.main --mode benchmark --config configs/default.yaml --instance small --seeds 1 --methods uniform_30 battery_threshold dwell_based_greedy
```

### Medium main experiment (multi-seed)
```bash
python -m src.main --mode benchmark --config configs/default.yaml --instance medium --seeds 1 2 3 --methods uniform_30 battery_threshold proposed
```

### Large scalability experiment
```bash
python -m src.main --mode benchmark --experiment scalability --config configs/default.yaml --instances large --seeds 1 2 --methods uniform_30 proposed
```

## Benchmark, ablation, scalability, sensitivity
```bash
# Benchmark
python -m src.main --mode benchmark --config configs/default.yaml --instance medium --seeds 1 2 3 --methods uniform_30 battery_threshold proposed

# Ablation
python -m src.main --mode ablation --config configs/default.yaml --instance medium --seeds 1 2 3

# Sensitivity analysis
python -m src.main --mode sensitivity --config configs/default.yaml --instance medium --seeds 1 2 --methods uniform_30 proposed --sensitivity demand_intensity_factor --values 0.75 1.0 1.25
```

## Export paper tables
```bash
python -m src.main --mode export_tables --config configs/default.yaml --experiment benchmark
```

## Output locations
- Generated instances/scenarios: `data/generated/<instance>/...`
- Offline assignments: `outputs/assignments/...`
- Evaluation metrics: `outputs/metrics/...`
- Experiment summaries: `outputs/results/<experiment>/<instance>/summary.csv`
- Exported tables: `outputs/tables/...`
- Run metadata snapshots: `outputs/runs/...`

## Reproducibility notes
- Use explicit `--seed` / `--seeds` for all runs.
- Keep config snapshots from `outputs/runs` with reported results.
- `pytest` now creates temporary generated/output directories for tests, so clean checkout testing does not require manual pre-generation.

## Known limitations
- Large experiments can be compute-intensive and time-consuming.
- RL performance can vary by seed and hardware.
- Table export expects required metrics columns; missing metrics are treated as errors.
- Some policy methods may require checkpoints unless `--train-if-missing` is enabled.

## Testing
- Unit/integration markers are provided: `unit`, `integration`, `slow`, `smoke`.
- Full test suite:
```bash
python -m pytest -q
```
