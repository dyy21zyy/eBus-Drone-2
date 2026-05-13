import numpy as np

from src.data_generation.scenario_generator import generate_instance
from src.offline.assignment_data_builder import build_assignment_data
from src.offline.assignment_solver import solve_assignment
from src.env.ebus_drone_env import EBusDroneEnv
from src.utils.config import load_yaml


def _build_env(seed=1):
    cfg = load_yaml('configs/default.yaml')
    inst_cfg = load_yaml('configs/instances/small.yaml')
    instance = generate_instance(cfg, inst_cfg, seed=seed)
    scenario = {"passenger": {"passenger_arrivals": {}}, "power": {"station_loads_kw": {}}}
    assignment = solve_assignment(build_assignment_data(instance)).to_dict()
    return EBusDroneEnv(config=cfg, instance=instance, scenario=scenario, assignment=assignment)


def test_observation_dimension_fixed_across_steps():
    env = _build_env(1)
    obs, _ = env.reset()
    d = obs.shape[0]
    for _ in range(3):
        obs, _, term, _, _ = env.step(0)
        assert obs.shape[0] == d
        if term:
            break


def test_observation_event_features_nonzero_when_event_has_values():
    env = _build_env(1)
    obs, _ = env.reset()
    names = env.observation_feature_names
    idxs = [names.index('l_alighting'), names.index('l_initial_board'), names.index('l_unloading_volume')]
    assert any(obs[i] > 0 for i in idxs)


def test_time_cyclic_encoding_changes_and_bounded():
    env = _build_env(1)
    obs0, _ = env.reset()
    i_sin = env.observation_feature_names.index('l_time_sin')
    i_cos = env.observation_feature_names.index('l_time_cos')
    obs1, _, _, _, _ = env.step(0)
    assert -1.0 <= obs0[i_sin] <= 1.0 and -1.0 <= obs0[i_cos] <= 1.0
    assert -1.0 <= obs1[i_sin] <= 1.0 and -1.0 <= obs1[i_cos] <= 1.0
    assert (obs0[i_sin] != obs1[i_sin]) or (obs0[i_cos] != obs1[i_cos])


def test_station_and_trip_identity_are_one_hot_not_scalar_ids():
    env = _build_env(1)
    obs, _ = env.reset()
    names = env.observation_feature_names
    s_idx = [i for i, n in enumerate(names) if n.startswith('l_station_onehot_')]
    t_idx = [i for i, n in enumerate(names) if n.startswith('l_trip_onehot_')]
    assert len(s_idx) > 1 and len(t_idx) > 1
    assert np.isclose(float(np.sum(obs[s_idx])), 1.0)
    assert np.isclose(float(np.sum(obs[t_idx])), 1.0)


def test_feature_name_length_matches_observation_dimension():
    env = _build_env(1)
    obs, _ = env.reset()
    assert len(env.observation_feature_names) == obs.shape[0]
