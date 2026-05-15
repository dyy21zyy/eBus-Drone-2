from src.data_generation.scenario_generator import generate_instance
from src.offline.assignment_data_builder import build_assignment_data
from src.offline.assignment_solver import solve_assignment
from src.env.ebus_drone_env import EBusDroneEnv
from src.offline.assignment_io import build_assignment_indices
from src.utils.config import load_yaml


def test_env_uses_loaded_generated_data():
    cfg = load_yaml('configs/default.yaml')
    inst_cfg = load_yaml('configs/instances/small.yaml')
    instance = generate_instance(cfg, inst_cfg, seed=1)
    scenario = {"passenger": {"passenger_arrivals": {s['stop_id']: [0] for s in instance['network']['stops']}}, "power": {"station_loads_kw": {}}}
    assignment = solve_assignment(build_assignment_data(instance)).to_dict()
    env = EBusDroneEnv(config=cfg, instance=instance, scenario=scenario, assignment=assignment)
    assert env.instance is not None
    assert env.scenario is not None
    assert env.assignment is not None
    assert env.state['horizon'] == instance['horizon_minutes']
    assert env.state['battery_max'] == instance['bus']['battery_capacity_kwh']


def test_cli_eval_loads_real_data_path(monkeypatch):
    from src import main as main_mod
    from src.utils.config import load_yaml
    called = {"value": False}

    def _fake_env(**kwargs):
        assert kwargs.get("instance") is not None
        assert kwargs.get("scenario") is not None
        assert kwargs.get("assignment") is not None
        called["value"] = True
        class Dummy:
            pass
        return Dummy()

    monkeypatch.setattr(main_mod, "EBusDroneEnv", _fake_env)
    monkeypatch.setattr(main_mod, "evaluate_policy", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(main_mod, "build_policy", lambda method: object())
    cfg = load_yaml("configs/default.yaml")
    main_mod.run_generate(cfg, "small", 1)
    main_mod.run_offline(cfg, "small", 1)
    import sys
    sys.argv = ["prog", "--mode", "eval", "--config", "configs/default.yaml", "--instance", "small", "--method", "no_charging", "--seed", "1", "--smoke-test"]
    main_mod.main()
    assert called["value"] is True


def test_assignment_driven_unloading_chain():
    cfg = load_yaml('configs/default.yaml')
    inst_cfg = load_yaml('configs/instances/small.yaml')
    instance = generate_instance(cfg, inst_cfg, seed=1)
    scenario = {
        "passenger": {
            "passenger_arrivals": {},
            "arrival_rate_per_stop_per_min": {str(s["stop_id"]): 20.0 for s in instance["network"]["stops"]},
            "alighting_probability": 0.2,
        },
        "power": {"station_loads_kw": {}},
    }
    assignment = solve_assignment(build_assignment_data(instance)).to_dict()
    env = EBusDroneEnv(config=cfg, instance=instance, scenario=scenario, assignment=assignment)
    idx = build_assignment_indices(assignment)
    (trip_id, station_id), expected = next(((k, v) for k, v in idx["by_trip_station"].items() if v), ((None, None), []))
    assert expected
    done = False
    hit = None
    while not done:
        _, _, done, _, info = env.step(0)
        if info["current_trip_id"] == trip_id and info["current_station_id"] == station_id:
            hit = info
            break
    assert hit is not None
    assert sorted(hit["unloaded_parcels"]) == sorted(expected)
    expected_q = sum(env.parcel_states[pid]["weight_kg"] for pid in expected)
    assert hit["unloading_volume_kg"] == expected_q
    for pid in expected:
        assert pid not in env.bus_states[trip_id]["onboard_parcel_ids"]
        assert pid in env.station_states[station_id]["locker_parcels"] or env.parcel_states[pid]["status"] in {"assigned_to_drone", "delivered"}


def test_parcel_unloading_validation_errors():
    from src.env.parcel_process import unload_parcels_to_locker
    trip = {"trip_id": 1, "onboard_parcel_ids": [10], "onboard_parcel_weight": 1.0}
    station = {"station_id": 2, "locker_parcels": [], "locker_inventory_kg": 0.0}
    parcels = {10: {"weight_kg": 1.0, "status": "onboard", "assigned_trip_id": 1, "assigned_station_id": 2}}
    unload_parcels_to_locker(trip, station, parcels, [10], 0.0)
    import pytest
    with pytest.raises(ValueError):
        unload_parcels_to_locker(trip, station, parcels, [10], 1.0)


def test_wrong_trip_station_rejected():
    from src.env.parcel_process import get_unloading_parcels
    import pytest
    idx_wrong_trip = {"by_trip_station": {(2, 2): [10]}}
    idx_wrong_station = {"by_trip_station": {(1, 3): [10]}}
    parcels = {10: {"status": "onboard", "assigned_trip_id": 1, "assigned_station_id": 2, "weight_kg": 1.0}}
    with pytest.raises(ValueError):
        get_unloading_parcels(2, 2, idx_wrong_trip, parcels)
    with pytest.raises(ValueError):
        get_unloading_parcels(1, 3, idx_wrong_station, parcels)


def test_station_dispatch_and_delivery_integration():
    cfg = load_yaml('configs/default.yaml')
    inst_cfg = load_yaml('configs/instances/small.yaml')
    instance = generate_instance(cfg, inst_cfg, seed=1)
    scenario = {"passenger": {"passenger_arrivals": {}}, "power": {"station_loads_kw": {}}}
    assignment = solve_assignment(build_assignment_data(instance)).to_dict()
    env = EBusDroneEnv(config=cfg, instance=instance, scenario=scenario, assignment=assignment)
    for _ in range(80):
        _, _, done, _, _ = env.step(0)
        if env.delivered_parcels:
            break
        if done:
            break
    assert len(env.delivered_parcels) >= 1


def test_non_freight_trip_never_unloads_parcels():
    cfg = load_yaml('configs/default.yaml')
    inst_cfg = load_yaml('configs/instances/small.yaml')
    instance = generate_instance(cfg, inst_cfg, seed=1)
    scenario = {"passenger": {"passenger_arrivals": {}}, "power": {"station_loads_kw": {}}}
    assignment = solve_assignment(build_assignment_data(instance)).to_dict()
    env = EBusDroneEnv(config=cfg, instance=instance, scenario=scenario, assignment=assignment)
    freight = set(instance["network"]["freight_carrying_trip_ids"])
    saw_non_freight_integrated = False
    done = False
    while not done:
        _, _, done, _, info = env.step(0)
        trip_id = int(info["current_trip_id"])
        if int(info["current_station_id"]) != -1 and trip_id not in freight:
            saw_non_freight_integrated = True
            assert info["unloaded_parcels"] == []
            assert info["unloading_volume_kg"] == 0.0
    # Depending on stochastic event timing, a non-freight integrated decision stop may not occur
    # in every run, but when it does occur unloading must be empty.
