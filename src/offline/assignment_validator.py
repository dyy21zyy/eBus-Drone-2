from __future__ import annotations

from collections import defaultdict

from src.offline.assignment_data_builder import AssignmentData
from src.offline.assignment_result import AssignmentResult


def validate_assignment(data: AssignmentData, result: AssignmentResult) -> None:
    by_customer = defaultdict(int)
    bus_load = defaultdict(float)
    unload_load = defaultdict(float)
    drone_work = defaultdict(float)
    locker_load = {(h, tau): 0.0 for h in data.stations for tau in data.t_grid}

    for d in result.decisions:
        key = (d.trip_id, d.station_id, d.customer_id)
        if d.station_id not in data.feasible_stations_by_customer[d.customer_id]:
            raise ValueError(f"Invalid assignment: customer {d.customer_id} not feasible for station {d.station_id}")
        by_customer[d.customer_id] += 1
        q = data.parcel_weight[d.customer_id]
        bus_load[d.trip_id] += q
        unload_load[(d.trip_id, d.station_id)] += q
        drone_work[d.station_id] += data.t_hi_rt[(d.station_id, d.customer_id)]
        r = data.r_bhi_0[key]
        p = data.p_bhi_0[key]
        for tau in data.t_grid:
            if r <= tau < p:
                locker_load[(d.station_id, tau)] += q

    for i in data.customers:
        if by_customer[i] != 1:
            raise ValueError(f"Constraint violated: customer {i} assigned {by_customer[i]} times")
    for b in data.trips:
        if bus_load[b] > data.q_f[b] + 1e-9:
            raise ValueError(f"Constraint violated: bus trip freight capacity for trip {b}")
    for b in data.trips:
        for h in data.stations:
            if unload_load[(b, h)] > data.q_u[h] + 1e-9:
                raise ValueError(f"Constraint violated: unloading capacity for trip {b}, station {h}")
    for h in data.stations:
        for tau in data.t_grid:
            if locker_load[(h, tau)] > data.k[h] + 1e-9:
                raise ValueError(f"Constraint violated: locker capacity for station {h}, tau={tau}")
    for h in data.stations:
        if drone_work[h] > data.num_drones[h] * data.operating_horizon + 1e-9:
            raise ValueError(f"Constraint violated: aggregate drone workload for station {h}")
