import json
from pathlib import Path
from src.utils.config import load_yaml
from src.data_generation.scenario_generator import generate_instance, generate_scenario
from src.offline.assignment_data_builder import build_assignment_data


def test_instance_constraints():
    cfg = load_yaml('configs/default.yaml')
    inst_cfg = load_yaml('configs/instances/small.yaml')
    instance = generate_instance(cfg, inst_cfg, seed=1)
    assert instance['network']['scheduled_bus_trips']
    assert instance['network']['stops']
    assert instance['stations']['stations']
    assert instance['customers']
    assert instance['network']['nominal_travel_time_min']


def test_generate_command_outputs():
    out = Path('data/generated/small')
    assert (out / 'instance_seed_1.json').exists()
    assert (out / 'scenario_0_seed_1.json').exists()
    sc = json.loads((out / 'scenario_0_seed_1.json').read_text())
    assert 'passenger' in sc and 'power' in sc


def test_generated_deadlines_have_planned_feasible_pair():
    cfg = load_yaml('configs/default.yaml')
    inst_cfg = load_yaml('configs/instances/small.yaml')
    instance = generate_instance(cfg, inst_cfg, seed=2)
    data = build_assignment_data(instance)
    for i in data.customers:
        feasible = False
        for h in data.feasible_stations_by_customer[i]:
            for b in data.trips:
                key = (b, h, i)
                if data.c_bhi_0[key] <= data.deadline[i] + 1e-9:
                    feasible = True
                    break
            if feasible:
                break
        assert feasible, f"customer {i} has no planned-feasible (trip, station) pair"
