from __future__ import annotations
import random, math


def generate_customers_and_parcels(config: dict, instance_cfg: dict, stops: list[dict], station_ids: list[int], seed: int) -> dict:
    rng = random.Random(seed + 17)
    area = config["network"]["service_area"]
    station_x = {s["stop_id"]: s["x_km"] for s in stops if s["stop_id"] in station_ids}
    values = list(config["parcel"]["weight_values_kg"])
    max_one_way = float(config["parcel"]["drone_round_trip_range_km"]) / 2.0
    speed = float(config["drone"]["speed_kmh"])
    svc = float(config["drone"]["customer_service_time_min"])
    turn = float(config["drone"]["turnaround_time_min"])
    max_dur = float(config["drone"]["max_round_trip_duration_min"])
    horizon = float(config["generation"]["horizon_minutes"])
    wait_nominal = float(config["parcel"]["nominal_locker_waiting_time_min"])
    nominal_unloading = float(config["parcel"].get("nominal_unloading_time_min", 0.0))
    allow_infeasible_deadlines = bool(config.get("generation", {}).get("allow_infeasible_deadlines", False))
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
                fastest_outbound = min(v["outbound_with_service_time_min"] for v in feasible)
                min_feasible_deadline = nominal_unloading + wait_nominal + fastest_outbound
                if allow_infeasible_deadlines:
                    deadline = rng.uniform(0.0, horizon)
                else:
                    if min_feasible_deadline > horizon:
                        continue
                    deadline = rng.uniform(min_feasible_deadline, horizon)
                customers.append({"customer_id":cid,"x_km":round(x,4),"y_km":round(y,4),"parcel_weight_kg":rng.choice(values),"feasible_stations":feasible,"delivery_deadline_min":round(deadline,4)})
                generated = True
                break
        if not generated:
            raise ValueError(
                f"Unable to generate feasible customer {cid} in {max_customer_retries} attempts; "
                f"allow_infeasible_deadlines={allow_infeasible_deadlines}."
            )
    return {"customers":customers}
