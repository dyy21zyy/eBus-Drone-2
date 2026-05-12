import json
from pathlib import Path
from src.main import run_generate
from src.utils.config import load_yaml


def _load_scenario(instance='small'):
    return json.loads((Path(f'data/generated/{instance}') / 'scenario_1.json').read_text())


def test_seed_reproducibility_and_variation():
    cfg = load_yaml('configs/default.yaml')
    run_generate(cfg, 'small', 7)
    sc1 = _load_scenario()
    run_generate(cfg, 'small', 7)
    sc2 = _load_scenario()
    run_generate(cfg, 'small', 8)
    sc3 = _load_scenario()
    assert sc1 == sc2
    assert sc1 != sc3
