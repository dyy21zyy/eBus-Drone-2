import numpy as np

from src.env.action_space import A_FULL, feasible_action_mask, repair_action
from src.env.bus_process import apply_travel_consumption, charged_energy_kwh
from src.data_generation.scenario_generator import generate_instance
from src.offline.assignment_data_builder import build_assignment_data
from src.offline.assignment_solver import solve_assignment
from src.env.ebus_drone_env import EBusDroneEnv
from src.rl.agents.am_ddqn_dr_agent import AMDDQNDRAgent
from src.utils.config import load_yaml


def test_no_charger_only_zero_feasible():
    mask = feasible_action_mask(available_chargers=0, current_battery_kwh=50, capacity_kwh=150, power_kw=500, eta=0.95)
    assert mask.tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 0]


def test_charged_energy_15_sec_500kw_eta_095():
    assert abs(charged_energy_kwh(500, 15, 0.95) - 1.9791666667) < 1e-8


def test_charged_energy_120_sec_500kw_eta_095():
    assert abs(charged_energy_kwh(500, 120, 0.95) - 15.8333333333) < 1e-8


def test_near_full_battery_masks_long_actions():
    mask = feasible_action_mask(available_chargers=1, current_battery_kwh=159.0, capacity_kwh=160.0, power_kw=500.0, eta=0.95)
    feasible_durations = [A_FULL[i] for i, v in enumerate(mask.tolist()) if v == 1]
    assert feasible_durations == [0]


def test_station_power_not_in_mask():
    mask1 = feasible_action_mask(1, 100, 150, 500, 0.95)
    mask2 = feasible_action_mask(1, 100, 150, 500, 0.95)
    assert np.array_equal(mask1, mask2)


def test_infeasible_repaired_to_feasible():
    mask = feasible_action_mask(1, 159, 160, 500, 0.95)
    repaired = repair_action(8, mask)
    assert mask[repaired] == 1
    assert repaired <= 8


def test_travel_consumption_10km_16kwh_per_km():
    assert apply_travel_consumption(100.0, 10.0, 1.6) == 84.0


def test_masked_agent_never_picks_infeasible_actions():
    obs = np.zeros(4, dtype=np.float32)
    mask = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    agent = AMDDQNDRAgent(4, 9, {"epsilon_start": 1.0, "epsilon_final": 1.0})
    for _ in range(200):
        a = agent.select_action(obs, mask, training=True)
        assert mask[a] == 1


def test_parcel_only_stop_is_decision_epoch():
    cfg = load_yaml('configs/default.yaml')
    inst_cfg = load_yaml('configs/instances/small.yaml')
    instance = generate_instance(cfg, inst_cfg, seed=1)
    scenario = {"passenger": {"passenger_arrivals": {}}, "power": {"station_loads_kw": {}}}
    assignment = solve_assignment(build_assignment_data(instance)).to_dict()
    env = EBusDroneEnv(config=cfg, instance=instance, scenario=scenario, assignment=assignment)
    assert env.current_event is not None
    assert env.current_event.parcel_required is True
