from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssignmentData:
    customers: list[int]
    trips: list[int]
    stations: list[int]
    station_stop_by_id: dict[int, int]
    feasible_stations_by_customer: dict[int, set[int]]
    feasible_customers_by_station: dict[int, list[int]]
    parcel_weight: dict[int, float]
    deadline: dict[int, float]
    q_f: dict[int, float]
    q_u: dict[int, float]
    k: dict[int, float]
    num_drones: dict[int, int]
    operating_horizon: float
    delta_bh: dict[tuple[int, int], int]
    c_b: dict[tuple[int, int], float]
    c_d: dict[tuple[int, int], float]
    beta_h: float
    beta_l: float
    t_bh_0: dict[tuple[int, int], float]
    t_u_bh_0: dict[tuple[int, int], float]
    r_bhi_0: dict[tuple[int, int, int], float]
    w_hi_d_0: dict[tuple[int, int], float]
    p_bhi_0: dict[tuple[int, int, int], float]
    t_hi_out: dict[tuple[int, int], float]
    t_hi_rt: dict[tuple[int, int], float]
    c_bhi_0: dict[tuple[int, int, int], float]
    lateness_0_plus: dict[tuple[int, int, int], float]
    h_bhi_0: dict[tuple[int, int, int], float]
    t_grid: list[int]


def build_assignment_data(instance: dict) -> AssignmentData:
    customers_raw = instance["customers"]
    trips_raw = instance["network"]["scheduled_bus_trips"]
    stations_raw = instance["stations"]["stations"]
    travel = instance["network"]["nominal_travel_time_min"]
    bus_cost_kgkm = float(instance["parcel"]["cost"]["bus_transport_per_kgkm"])
    drone_cost_km = float(instance["parcel"]["cost"]["drone_per_km"])
    beta_h = float(instance["parcel"]["cost"]["locker_holding_per_kgmin"])
    beta_l = float(instance["parcel"]["cost"]["lateness_per_min"])
    wait_nominal = float(instance["parcel"]["nominal_locker_waiting_time_min"])
    unload_sec_per_kg = float(instance["parcel"]["unloading_time_sec_per_kg"])
    nominal_unloading_time_min = float(instance["parcel"].get("nominal_unloading_time_min", (instance["parcel"]["unloading_capacity_kg_per_stop"] * unload_sec_per_kg) / 60.0))

    customers = [int(c["customer_id"]) for c in customers_raw]
    trips = [int(b["trip_id"]) for b in trips_raw]
    stations = [int(s["station_id"]) for s in stations_raw]
    stop_ids = [int(s["stop_id"]) for s in instance["network"]["stops"]]
    stop_index_by_id = {sid: idx for idx, sid in enumerate(stop_ids)}
    station_stop_by_id = {int(s["station_id"]): int(s.get("stop_id", s["station_id"])) for s in stations_raw}

    station_by_id = {int(s["station_id"]): s for s in stations_raw}
    trip_departure = {int(b["trip_id"]): float(b["departure_min"]) for b in trips_raw}

    parcel_weight = {int(c["customer_id"]): float(c["parcel_weight_kg"]) for c in customers_raw}
    deadline = {int(c["customer_id"]): float(c["delivery_deadline_min"]) for c in customers_raw}

    feasible_stations_by_customer: dict[int, set[int]] = {}
    feasible_customers_by_station: dict[int, list[int]] = {h: [] for h in stations}
    t_hi_out: dict[tuple[int, int], float] = {}
    t_hi_rt: dict[tuple[int, int], float] = {}
    c_d: dict[tuple[int, int], float] = {}

    for c in customers_raw:
        i = int(c["customer_id"])
        feasible = set()
        for option in c["feasible_stations"]:
            h = int(option["station_id"])
            feasible.add(h)
            feasible_customers_by_station[h].append(i)
            d = float(option["distance_km"])
            outbound = float(option["outbound_with_service_time_min"])
            rt = float(option["mission_duration_min"])
            t_hi_out[(h, i)] = outbound
            t_hi_rt[(h, i)] = rt
            c_d[(h, i)] = drone_cost_km * d
        feasible_stations_by_customer[i] = feasible
        if not feasible:
            raise ValueError(f"Customer {i} has no feasible station candidates.")

    q_f = {b: float(instance["bus"]["freight_capacity_kg"]) for b in trips}
    q_u = {h: float(instance["parcel"]["unloading_capacity_kg_per_stop"]) for h in stations}
    k = {h: float(station_by_id[h]["locker_capacity_kg"]) for h in stations}
    num_drones = {h: int(station_by_id[h]["drones"]) for h in stations}
    operating_horizon = float(instance["horizon_minutes"])

    delta_bh: dict[tuple[int, int], int] = {}
    c_b: dict[tuple[int, int], float] = {}
    t_bh_0: dict[tuple[int, int], float] = {}
    t_u_bh_0: dict[tuple[int, int], float] = {}

    for b in trips:
        dep = trip_departure[b]
        for h in stations:
            stop_id = station_stop_by_id[h]
            if stop_id not in stop_index_by_id:
                raise ValueError(f"Station {h} references unknown stop_id={stop_id}")
            stop_idx = stop_index_by_id[stop_id]
            # On the default corridor route every trip visits every stop; this explicitly maps station->stop.
            delta_bh[(b, h)] = 1 if stop_idx < len(travel[0]) else 0
            station_dist_km = float(instance["network"]["distances_km"][0][stop_idx])
            c_b[(b, h)] = bus_cost_kgkm * station_dist_km
            t_bh_0[(b, h)] = dep + float(travel[0][stop_idx])
            # T_bh_U_0 nominal unloading time for bus-trip/station event.
            # We keep this deterministic and configurable; by default we preserve the
            # prior conservative approximation derived from unloading capacity * sec/kg.
            t_u_bh_0[(b, h)] = nominal_unloading_time_min

    r_bhi_0 = {}
    p_bhi_0 = {}
    c_bhi_0 = {}
    lateness_0_plus = {}
    h_bhi_0 = {}
    w_hi_d_0 = {}
    for h in stations:
        for i in feasible_customers_by_station[h]:
            w_hi_d_0[(h, i)] = wait_nominal
            for b in trips:
                r = t_bh_0[(b, h)] + t_u_bh_0[(b, h)]
                p = r + wait_nominal
                completion = p + t_hi_out[(h, i)]
                key = (b, h, i)
                r_bhi_0[key] = r
                p_bhi_0[key] = p
                c_bhi_0[key] = completion
                lateness_0_plus[key] = max(0.0, completion - deadline[i])
                h_bhi_0[key] = p - r

    t_grid = list(range(int(operating_horizon) + 1))

    return AssignmentData(
        customers=customers, trips=trips, stations=stations,
        station_stop_by_id=station_stop_by_id,
        feasible_stations_by_customer=feasible_stations_by_customer,
        feasible_customers_by_station=feasible_customers_by_station,
        parcel_weight=parcel_weight, deadline=deadline, q_f=q_f, q_u=q_u, k=k,
        num_drones=num_drones, operating_horizon=operating_horizon, delta_bh=delta_bh,
        c_b=c_b, c_d=c_d, beta_h=beta_h, beta_l=beta_l, t_bh_0=t_bh_0, t_u_bh_0=t_u_bh_0,
        r_bhi_0=r_bhi_0, w_hi_d_0=w_hi_d_0, p_bhi_0=p_bhi_0, t_hi_out=t_hi_out, t_hi_rt=t_hi_rt,
        c_bhi_0=c_bhi_0, lateness_0_plus=lateness_0_plus, h_bhi_0=h_bhi_0, t_grid=t_grid
    )
