import json
from pathlib import Path
from src.main import run_generate
from src.utils.config import load_yaml


def test_generation_outputs_and_feasibility(tmp_path):
    cfg = load_yaml('configs/default.yaml')
    run_generate(cfg, 'small', 1)
    base = Path('data/generated/small')
    inst = json.loads((base / 'instance.json').read_text())
    sc = json.loads((base / 'scenario_1.json').read_text())
    assert len(inst['customers']) == 30
    assert len(inst['stations']) == 6
    assert len(inst['bus_trips']) == 24
    assert sc['seed'] == 1
    assert len(sc['passenger_arrivals']) == inst['network']['num_stops']
    for c in inst['customers']:
        assert c['feasible_stations']
        assert c['deadline_min'] <= cfg['network']['horizon_minutes']
