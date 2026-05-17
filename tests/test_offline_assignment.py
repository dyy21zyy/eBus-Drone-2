import json
from pathlib import Path
from dataclasses import replace

from src.offline.assignment_data_builder import build_assignment_data
from src.offline.assignment_io import read_assignment, write_assignment
from src.offline.assignment_io import build_assignment_indices
from src.offline.assignment_solver import solve_assignment
from src.offline.assignment_solver import AssignmentInfeasibleError
from src.main import run_generate


def _load_instance(seed: int = 1, instance: str = "small"):
    fp = Path("data/generated") / instance / f"instance_seed_{seed}.json"
    if not fp.exists():
        from src.utils.config import load_yaml
        run_generate(load_yaml("configs/default.yaml"), instance, seed)
    assert fp.exists(), "Run generate mode first"
    return json.loads(fp.read_text(encoding="utf-8"))


def test_offline_assignment_constraints_and_io(tmp_path):
    instance = _load_instance()
    data = build_assignment_data(instance)
    result = solve_assignment(data, allow_greedy_fallback=False)

    assert len(result.decisions) == len(data.customers)
    assert len({d.customer_id for d in result.decisions}) == len(data.customers)

    bus_load = {b: 0.0 for b in data.trips}
    unload = {(b, h): 0.0 for b in data.trips for h in data.stations}
    drone = {h: 0.0 for h in data.stations}
    locker_seen = False
    locker_load = {(h, tau): 0.0 for h in data.stations for tau in data.t_grid}

    for d in result.decisions:
        q = data.parcel_weight[d.customer_id]
        bus_load[d.trip_id] += q
        unload[(d.trip_id, d.station_id)] += q
        drone[d.station_id] += data.t_hi_rt[(d.station_id, d.customer_id)]
        key = (d.trip_id, d.station_id, d.customer_id)
        for tau in data.t_grid:
            if data.r_bhi_0[key] <= tau < data.p_bhi_0[key]:
                locker_load[(d.station_id, tau)] += q
                locker_seen = True

    assert locker_seen
    assert all(bus_load[b] <= data.q_f[b] + 1e-9 for b in data.trips)
    assert all(unload[(b, h)] <= data.q_u[h] + 1e-9 for b in data.trips for h in data.stations)
    assert all(locker_load[(h, tau)] <= data.k[h] + 1e-9 for h in data.stations for tau in data.t_grid)
    assert all(drone[h] <= data.num_drones[h] * data.operating_horizon + 1e-9 for h in data.stations)
    assert result.status
    assert result.objective_value is not None
    assert result.solver_name
    assert result.number_assigned_customers == len(data.customers)
    assert result.number_customers == len(data.customers)
    assert isinstance(result.used_fallback, bool)

    out = tmp_path / "assignment.json"
    write_assignment(result, str(out))
    loaded = read_assignment(str(out))
    assert loaded.total_cost == result.total_cost
    assert len(loaded.decisions) == len(result.decisions)
    assert "metadata" not in loaded.to_dict() or isinstance(loaded.to_dict().get("metadata"), (dict, type(None)))


def test_nominal_drone_timing_contains_service_and_turnaround():
    instance = _load_instance()
    data = build_assignment_data(instance)
    svc = float(instance["drone"]["customer_service_time_min"])
    turn = float(instance["drone"]["turnaround_time_min"])
    speed = float(instance["drone"]["speed_kmh"])
    for c in instance["customers"]:
        i = int(c["customer_id"])
        for option in c["feasible_stations"]:
            h = int(option["station_id"])
            d = float(option["distance_km"])
            expected_out = (d / speed) * 60.0 + svc
            expected_rt = (2.0 * d / speed) * 60.0 + svc + turn
            assert abs(data.t_hi_out[(h, i)] - expected_out) < 1e-3
            assert abs(data.t_hi_rt[(h, i)] - expected_rt) < 1e-3


def test_build_assignment_indices_roundtrip():
    instance = _load_instance()
    data = build_assignment_data(instance)
    result = solve_assignment(data, allow_greedy_fallback=False).to_dict()
    idx = build_assignment_indices(result)
    assert idx["by_trip_station"]
    assert len(idx["by_customer"]) == len(data.customers)


def test_no_silent_fallback():
    instance = _load_instance()
    data0 = build_assignment_data(instance)
    q_f = dict(data0.q_f)
    for b in data0.trips:
        q_f[b] = 0.0
    data = replace(data0, q_f=q_f)
    try:
        solve_assignment(data, allow_greedy_fallback=False)
    except AssignmentInfeasibleError:
        pass
    else:
        raise AssertionError("Expected AssignmentInfeasibleError when fallback is disabled")


def test_assignment_output_metadata():
    from src.main import run_offline
    from src.utils.config import load_yaml

    cfg = load_yaml("configs/default.yaml")
    run_offline(cfg, "small", 1)
    payload = _load_instance()
    assert payload
    out = Path(cfg["paths"]["outputs"]) / "assignments" / "offline_assignment_small_seed_1.json"
    result_payload = json.loads(out.read_text(encoding="utf-8"))
    meta = result_payload.get("metadata", {})
    assert meta["instance_name"] == "small"
    assert meta["seed"] == 1
    assert "unloaded_parcel_volume_kg_by_bus_station" in meta
    assert "planned_locker_occupancy_summary_kg" in meta
    assert "planned_drone_workload_by_station_min" in meta
    assert "objective_value" in meta and meta["objective_value"] is not None
    assert "solver_status" in meta and meta["solver_status"]
    assert "bus_load_kg_by_bus" in meta and meta["bus_load_kg_by_bus"]
    assert "planned_drone_workload_summary_min" in meta


def test_station_id_can_differ_from_stop_id_for_arrival_mapping():
    instance = _load_instance()
    instance["stations"]["stations"][0]["station_id"] = 101
    instance["stations"]["stations"][0]["stop_id"] = 1
    old_id = instance["stations"]["station_ids"][0]
    instance["stations"]["station_ids"][0] = 101
    for c in instance["customers"]:
        for opt in c["feasible_stations"]:
            if opt["station_id"] == old_id:
                opt["station_id"] = 101
    data = build_assignment_data(instance)
    trip_id = data.trips[0]
    dep = instance["network"]["scheduled_bus_trips"][0]["departure_min"]
    expected = dep + instance["network"]["nominal_travel_time_min"][0][0]
    assert abs(data.t_bh_0[(trip_id, 101)] - expected) < 1e-9


def test_generation_records_deadline_repair_metadata():
    from src.utils.config import load_yaml
    from src.data_generation.scenario_generator import generate_instance
    cfg = load_yaml("configs/default.yaml")
    inst_cfg = load_yaml("configs/instances/small.yaml")
    generated = generate_instance(cfg, inst_cfg, 1)
    meta = generated.get("generation_metadata", {})
    assert meta.get("deadline_repair_policy") == "paper_slack_from_earliest_planned_completion"
    assert "deadline_repaired" in meta


def test_assignment_data_uses_only_freight_carrying_trips():
    instance = _load_instance()
    freight = [int(x) for x in instance["network"]["freight_carrying_trip_ids"]]
    data = build_assignment_data(instance)
    assert data.trips == freight


def test_assignment_data_backward_compat_without_freight_trip_ids():
    import warnings
    instance = _load_instance()
    instance["network"].pop("freight_carrying_trip_ids", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = build_assignment_data(instance)
    assert data.trips == [int(t["trip_id"]) for t in instance["network"]["scheduled_bus_trips"]]
    assert any("missing network.freight_carrying_trip_ids" in str(w.message) for w in caught)
