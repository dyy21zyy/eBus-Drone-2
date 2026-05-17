from __future__ import annotations
import random, math


DEADLINE_CLASS_MIX = (
    ("tight", 0.30, (20.0, 40.0)),
    ("moderate", 0.50, (40.0, 80.0)),
    ("loose", 0.20, (80.0, 140.0)),
)


def _draw_deadline_class(rng: random.Random) -> tuple[str, tuple[float, float]]:
    u = rng.random()
    cdf = 0.0
    for name, prob, bounds in DEADLINE_CLASS_MIX:
        cdf += prob
        if u <= cdf:
            return name, bounds
    return DEADLINE_CLASS_MIX[-1][0], DEADLINE_CLASS_MIX[-1][2]


def generate_customers_and_parcels(
    config: dict,
    instance_cfg: dict,
    stops: list[dict],
    station_ids: list[int],
    scheduled_bus_trips: list[dict],
    freight_carrying_trip_ids: list[int],
    nominal_travel_time_min: list[list[float]],
    seed: int,
) -> dict:
    rng = random.Random(seed + 17)
    area = config["network"]["service_area"]
    station_x = {s["stop_id"]: s["x_km"] for s in stops if s["stop_id"] in station_ids}
    values = list(config["parcel"]["weight_values_kg"])
    parcel_cfg = config["parcel"]
    max_one_way = float(
        parcel_cfg.get(
            "drone_service_radius_km",
            parcel_cfg.get("drone_round_trip_range_km", 8.0),
        )
    )
    speed = float(config["drone"]["speed_kmh"])
    svc = float(config["drone"]["customer_service_time_min"])
    turn = float(config["drone"]["turnaround_time_min"])
    max_dur = float(config["drone"]["max_round_trip_duration_min"])
    trip_departure_by_id = {int(t["trip_id"]): float(t["departure_min"]) for t in scheduled_bus_trips}
    freight_trip_ids = [int(tid) for tid in freight_carrying_trip_ids]
    wait_nominal = float(config["parcel"]["nominal_locker_waiting_time_min"])
    nominal_unloading = float(config["parcel"].get("nominal_unloading_time_min", 0.0))
    max_customer_retries = int(config.get("generation", {}).get("max_customer_generation_retries", 100))
    customers=[]
    for cid in range(1, int(instance_cfg["num_customers"]) + 1):
        generated = False
        for _attempt in range(max_customer_retries):
            x = rng.uniform(float(area["x_min_km"]), float(area["x_max_km"]))
            y = rng.uniform(float(area["y_min_km"]), float(area["y_max_km"]))
            feasible=[]
            for sid,sx in station_x.items():
                d=math.dist((x,y),(sx,0.0))
                if d <= max_one_way:
                    outbound_with_service = (d / speed) * 60.0 + svc
                    rt=(2*d/speed)*60.0 + svc + turn
                    if rt <= max_dur:
                        feasible.append({"station_id":sid,"distance_km":round(d,4),"mission_duration_min":round(rt,4),"outbound_with_service_time_min":round(outbound_with_service,4)})
            if feasible:
                earliest_completion = float("inf")
                feasible_station_ids: list[int] = []
                for opt in feasible:
                    sid = int(opt["station_id"])
                    if sid not in feasible_station_ids:
                        feasible_station_ids.append(sid)
                    stop_idx = sid - 1
                    for tid in freight_trip_ids:
                        dep = trip_departure_by_id[tid]
                        arrival = dep + float(nominal_travel_time_min[0][stop_idx])
                        completion = arrival + nominal_unloading + wait_nominal + float(opt["outbound_with_service_time_min"])
                        earliest_completion = min(earliest_completion, completion)

                deadline_class, slack_bounds = _draw_deadline_class(rng)
                slack_min = rng.uniform(slack_bounds[0], slack_bounds[1])
                deadline = earliest_completion + slack_min
                customers.append({
                    "customer_id": cid,
                    "x_km": round(x, 4),
                    "y_km": round(y, 4),
                    "parcel_weight_kg": rng.choice(values),
                    "feasible_stations": feasible,
                    "feasible_station_ids": sorted(feasible_station_ids),
                    "earliest_planned_completion_min": round(earliest_completion, 4),
                    "deadline_class": deadline_class,
                    "delivery_deadline_min": round(deadline, 4),
                })
                generated = True
                break
        if not generated:
            raise ValueError(
                f"Unable to generate feasible customer {cid} in {max_customer_retries} attempts; "
                f"freight_carrying_trip_ids={freight_trip_ids}."
            )
    return {"customers":customers}
