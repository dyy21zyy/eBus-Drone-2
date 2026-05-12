import json
from pathlib import Path

from src.offline.assignment_data_builder import build_assignment_data
from src.offline.assignment_io import read_assignment, write_assignment
from src.offline.assignment_solver import solve_assignment


def _load_instance(seed: int = 1, instance: str = "small"):
    fp = Path("data/generated") / instance / f"instance_seed_{seed}.json"
    assert fp.exists(), "Run generate mode first"
    return json.loads(fp.read_text(encoding="utf-8"))


def test_offline_assignment_constraints_and_io(tmp_path):
    instance = _load_instance()
    data = build_assignment_data(instance)
    result = solve_assignment(data)

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

    out = tmp_path / "assignment.json"
    write_assignment(result, str(out))
    loaded = read_assignment(str(out))
    assert loaded.total_cost == result.total_cost
    assert len(loaded.decisions) == len(result.decisions)
