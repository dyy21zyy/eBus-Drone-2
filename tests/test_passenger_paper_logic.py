import numpy as np
from src.data_generation.scenario_generator import generate_instance, generate_scenario
from src.env.ebus_drone_env import EBusDroneEnv
from src.offline.assignment_data_builder import build_assignment_data
from src.offline.assignment_solver import solve_assignment
from src.utils.config import load_yaml


def _build(seed=1):
    cfg = load_yaml('configs/default.yaml')
    inst_cfg = load_yaml('configs/instances/small.yaml')
    instance = generate_instance(cfg, inst_cfg, seed=seed)
    scenario = generate_scenario(cfg, instance, seed, 0)
    assignment = solve_assignment(build_assignment_data(instance)).to_dict()
    return cfg, instance, scenario, assignment


def test_alighting_profile_stop_time_dependent():
    _, _, scenario, _ = _build(3)
    prof = scenario['passenger']['alighting_profile_per_stop']
    keys = list(prof.keys())
    assert len(keys) >= 2
    assert prof[keys[0]][0] != prof[keys[1]][0] or prof[keys[0]][0] != prof[keys[0]][1]


def test_scenario_reproducible_under_seed_for_arrival_profiles():
    cfg, instance, _, _ = _build(4)
    a = generate_scenario(cfg, instance, 4, 0)
    b = generate_scenario(cfg, instance, 4, 0)
    assert a['passenger']['arrival_rate_profile_per_stop_per_min'] == b['passenger']['arrival_rate_profile_per_stop_per_min']


def test_integrated_no_pax_no_parcel_skips_decision():
    cfg, instance, scenario, assignment = _build(5)
    scenario['passenger']['arrival_rate_per_stop_per_min'] = {str(s['stop_id']): 0.0 for s in instance['network']['stops']}
    scenario['passenger']['arrival_rate_profile_per_stop_per_min'] = {str(s['stop_id']): [0.0]*int(cfg['generation']['bus_operation_horizon_minutes']) for s in instance['network']['stops']}
    scenario['passenger']['alighting_probability'] = 0.0
    scenario['passenger']['alighting_profile_per_stop'] = {str(s['stop_id']): [0.0]*int(cfg['generation']['bus_operation_horizon_minutes']) for s in instance['network']['stops']}
    env = EBusDroneEnv(config=cfg, instance=instance, scenario=scenario, assignment=assignment)
    env.freight_carrying_trip_ids = set()
    env._advance_until_decision()
    ev = env.current_decision_event
    assert ev is None or (ev.passengers_required or ev.parcel_required)


def test_capacity_never_exceeded_with_high_demand():
    cfg, instance, scenario, assignment = _build(6)
    env = EBusDroneEnv(config=cfg, instance=instance, scenario=scenario, assignment=assignment)
    for sid in env.stop_ids:
        scenario['passenger']['arrival_rate_per_stop_per_min'][str(sid)] = 50.0
    for _ in range(20):
        _, _, done, _, _ = env.step(0)
        for b in env.bus_states.values():
            assert b['onboard_passengers'] <= b['passenger_capacity']
        if done:
            break
