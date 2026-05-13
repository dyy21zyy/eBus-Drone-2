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


def test_passenger_service_applied_once_at_decision_event(monkeypatch):
    env = _build_env(seed=1)
    ev = env.current_decision_event
    assert ev is not None
    bus = env.bus_states[ev.trip_id]
    stop_id = env.stop_ids[int(ev.stop_index)]
    q_before = env.stop_queues[stop_id]
    onboard_before = bus["onboard_passengers"]
    out = env.step(0)
    _ = out
    assert bus["onboard_before_service"] == onboard_before
    assert env.stop_queues[stop_id] != q_before or out[4]["passenger_service"]["total_board"] >= 0


def test_unloading_time_uses_config_seconds_per_kg():
    env = _build_env(seed=1)
    ev = env.current_decision_event
    assert ev is not None
    _, _, _, _, info = env.step(0)
    qf = info["unloading_volume_kg"]
    expected = qf * (env.instance["parcel"]["unloading_time_sec_per_kg"] / 60.0)
    assert info["unloading_duration_min"] == expected


def test_parcel_only_stop_triggers_decision_event():
    env = _build_env(seed=1)
    ev = env.current_decision_event
    assert ev is not None
    assert ev.parcel_required is True


def test_no_charger_event_feasible_action_only_zero():
    env = _build_env(seed=1)
    ev = env.current_decision_event
    assert ev is not None
    st = env.station_states[ev.station_id]
    st["charger_release_times_min"] = [env.state["time"] + 1000.0 for _ in st["charger_release_times_min"]]
    mask = env.get_action_mask().tolist()
    assert mask[0] == 1
    assert sum(mask) == 1


def test_onboard_parcel_load_decreases_by_unloaded_volume():
    env = _build_env(seed=1)
    ev = env.current_decision_event
    assert ev is not None
    bus = env.bus_states[ev.trip_id]
    before = sum(env.parcel_states[p]["weight_kg"] for p in bus["onboard_parcel_ids"])
    _, _, _, _, info = env.step(0)
    after = sum(env.parcel_states[p]["weight_kg"] for p in bus["onboard_parcel_ids"])
    assert before - after == info["unloading_volume_kg"]


def test_locker_inventory_increases_on_unload():
    env = _build_env(seed=1)
    ev = env.current_decision_event
    assert ev is not None
    st = env.station_states[ev.station_id]
    before = st["locker_inventory_kg"]
    _, _, _, _, info = env.step(0)
    assert st["locker_inventory_kg"] >= before
    for pid in info["unloaded_parcels"]:
        assert env.parcel_states[pid]["release_time_min"] == info["departure_time_min"]

def test_step_reports_transition_reward_deltas_and_components():
    env = _build_env(seed=3)
    ev = env.current_decision_event
    assert ev is not None
    before_t = env.state["time"]
    _, reward, _, _, info = env.step(0)
    assert info["transition_start_time"] == before_t
    assert info["transition_end_time"] >= info["departure_time_min"]
    assert "reward_components" in info and info["reward_components"]
    rc = info["reward_components"]
    alphas = env._reward_alphas()
    expected_cost = (
        alphas["alpha_1"] * info["passenger_delay_delta"]
        + alphas["alpha_2"] * (info["parcel_lateness_delta"] + info["terminal_undelivered_penalty"])
        + alphas["alpha_3"] * info["energy_consumption_delta"]
        + alphas["alpha_4"] * info["power_overload_delta"]
        + alphas["alpha_5"] * info["battery_violation_delta"]
        + alphas["alpha_6"] * info["locker_overflow_delta"]
    )
    assert reward == -expected_cost
    assert reward == rc["reward"]


def test_terminal_penalty_only_on_terminal_transition():
    env = _build_env(seed=4)
    penalty_seen = []
    for _ in range(200):
        _, _, done, _, info = env.step(0)
        penalty_seen.append(info.get("terminal_undelivered_penalty", 0.0))
        if done:
            break
    assert sum(1 for p in penalty_seen if p > 0.0) <= 1
    if penalty_seen:
        assert penalty_seen[-1] >= 0.0


def test_observed_passenger_event_matches_step_execution():
    env = _build_env(seed=11)
    ev = env.current_decision_event
    assert ev is not None
    preview = ev.passenger_service_preview
    _, _, _, _, info = env.step(0)
    service = info["passenger_service"]
    assert int(preview["alighting"]) == int(service["alighting"])
    assert int(preview["initial_board"]) == int(service["initial_board"])


def test_step_does_not_resample_initial_passenger_service(monkeypatch):
    env = _build_env(seed=12)
    ev = env.current_decision_event
    assert ev is not None
    import src.env.passenger_process as pp
    calls = {"n": 0}
    orig = pp.sample_alighting
    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)
    monkeypatch.setattr(pp, "sample_alighting", wrapped)
    monkeypatch.setattr(env, "_advance_until_decision", lambda: None)
    env.step(0)
    assert calls["n"] == 0


def test_additional_arrivals_during_dwell_can_extend_dwell():
    env = _build_env(seed=13)
    env.scenario.setdefault("passenger", {})["arrival_rate_per_stop_per_min"] = {str(s): 30.0 for s in env.stop_ids}
    ev = env.current_decision_event
    assert ev is not None
    base = float(ev.passenger_service_preview["passenger_dwell_min"])
    _, _, _, _, info = env.step(3)
    service = info["passenger_service"]
    assert service["board_during_normal"] + service["board_during_excess"] >= 0
    assert info["dwell_components"]["realized_dwell_min"] >= base


def test_stop_indicator_consistent_with_stored_event_and_unloading():
    env = _build_env(seed=14)
    ev = env.current_decision_event
    assert ev is not None
    expected = bool(ev.passenger_service_preview.get("chi", False) or ev.unloading_volume_kg > 0.0)
    assert ev.requires_stop is expected
    _, _, _, _, info = env.step(0)
    realized = bool(info["passenger_service"]["chi"] or info["unloading_volume_kg"] > 0.0)
    assert realized is expected
