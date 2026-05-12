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
    customers=[]
    for cid in range(1, int(instance_cfg["num_customers"]) + 1):
        while True:
            x = rng.uniform(float(area["x_min_km"]), float(area["x_max_km"]))
            y = rng.uniform(float(area["y_min_km"]), float(area["y_max_km"]))
            feasible=[]
            for sid,sx in station_x.items():
                d=math.dist((x,y),(sx,0.0))
                if d <= max_one_way:
                    rt=(2*d/speed)*60.0 + svc + turn
                    if rt <= max_dur:
                        feasible.append({"station_id":sid,"distance_km":round(d,4),"mission_duration_min":round(rt,4)})
            if feasible:
                fastest=min(v["mission_duration_min"] for v in feasible)
                deadline=rng.uniform(fastest+5.0,horizon)
                customers.append({"customer_id":cid,"x_km":round(x,4),"y_km":round(y,4),"parcel_weight_kg":rng.choice(values),"feasible_stations":feasible,"delivery_deadline_min":round(deadline,4)})
                break
    return {"customers":customers}
