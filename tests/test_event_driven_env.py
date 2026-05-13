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


def test_positive_charge_delays_downstream_arrival():
    env0 = _build_env(); env1 = _build_env()
    _, _, _, _, i0 = env0.step(0)
    _, _, _, _, i1 = env1.step(2)
    b0 = env0.bus_states[i0["current_trip_id"]]["next_arrival_time_min"]
    b1 = env1.bus_states[i1["current_trip_id"]]["next_arrival_time_min"]
    assert b1 >= b0


def test_no_decision_event_without_passenger_or_parcel_stop():
    env = _build_env()
    ev = env.current_decision_event
    assert ev is None or (ev.passengers_required or ev.parcel_required)


def test_buses_have_independent_batteries():
    env = _build_env()
    keys = sorted(env.bus_states)
    if len(keys) < 2:
        return
    env.bus_states[keys[0]]["battery_kwh"] -= 10.0
    assert env.bus_states[keys[0]]["battery_kwh"] != env.bus_states[keys[1]]["battery_kwh"]


def test_charger_occupation_persists_until_release():
    env = _build_env()
    ev = env.current_decision_event
    if ev is None:
        return
    st = env.station_states[ev.station_id]
    before = env._available_chargers(st, env.state["time"])
    env.step(2)
    after = env._available_chargers(st, env.state["time"])
    assert after <= before


def test_reset_step_compatibility_shape():
    env = _build_env()
    obs, info = env.reset()
    assert obs is not None
    assert "event" in info
    out = env.step(0)
    assert len(out) == 5

def test_no_valid_stop_no_decision_event_when_no_queue_no_unloading():
    env = _build_env(seed=1)
    env.scenario["passenger"]["arrival_rate_per_stop_per_min"] = {str(s): 0.0 for s in env.stop_ids}
    env.scenario["passenger"]["alighting_probability"] = 0.0
    env.stop_queues = {s: 0 for s in env.stop_ids}
    env._advance_until_decision()
    ev = env.current_decision_event
    assert ev is None or (ev.passengers_required or ev.parcel_required)


def test_ordinary_stop_updates_passengers_without_decision():
    env = _build_env(seed=2)
    env.scenario["passenger"]["arrival_rate_per_stop_per_min"] = {str(s): 2.0 for s in env.stop_ids}
    env.scenario["passenger"]["alighting_probability"] = 0.1
    # force process first event directly
    e = env.calendar.pop_next()
    pre = env.bus_states[e.trip_id]["onboard_passengers"]
    is_decision = env._process_bus_arrival(e)
    post = env.bus_states[e.trip_id]["onboard_passengers"]
    if not e.integrated:
        assert is_decision is False
    assert post >= 0 and pre >= 0
