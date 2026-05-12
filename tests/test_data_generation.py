import json
from pathlib import Path
from src.utils.config import load_yaml
from src.data_generation.scenario_generator import generate_instance, generate_scenario


def test_instance_constraints():
    cfg = load_yaml('configs/default.yaml')
    inst_cfg = load_yaml('configs/instances/small.yaml')
    instance = generate_instance(cfg, inst_cfg, seed=1)
    assert len(instance['network']['stops']) == 30
    assert len(instance['stations']['station_ids']) == 6
    assert len(instance['network']['scheduled_bus_trips']) == 24
    assert len(instance['customers']) == 30
    for c in instance['customers']:
        assert c['feasible_stations']
        assert c['delivery_deadline_min'] <= cfg['generation']['horizon_minutes']


def test_generate_command_outputs():
    out = Path('data/generated/small')
    assert (out / 'instance_seed_1.json').exists()
    assert (out / 'scenario_0_seed_1.json').exists()
    sc = json.loads((out / 'scenario_0_seed_1.json').read_text())
    assert 'passenger' in sc and 'power' in sc
