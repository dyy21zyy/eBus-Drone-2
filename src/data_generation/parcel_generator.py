import random


def _dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5


def generate_customers(num_customers: int, stop_positions: list[float], station_ids: list[int], cfg: dict, rng: random.Random) -> list[dict]:
    half_w = cfg["parcel"]["service_area_half_width_km"]
    max_one_way = cfg["parcel"]["drone_range_km_round_trip"] / 2.0
    weights = cfg["parcel"]["weight_values_kg"]
    station_xy = {sid: (stop_positions[sid-1], 0.0) for sid in station_ids}
    customers = []
    for i in range(1, num_customers + 1):
        while True:
            x = rng.uniform(0.0, stop_positions[-1])
            y = rng.uniform(-half_w, half_w)
            feasible = [sid for sid,xy in station_xy.items() if _dist((x,y),xy) <= max_one_way]
            if feasible:
                ddl = rng.randint(60, cfg["network"]["horizon_minutes"])
                nearest = min(_dist((x,y), station_xy[s]) for s in feasible)
                mission_min = (2*nearest / cfg["drone"]["speed_kmh"])*60 + cfg["drone"]["service_time_min"] + cfg["drone"]["turnaround_time_min"]
                if ddl >= mission_min:
                    customers.append({"customer_id": i, "x_km": x, "y_km": y, "parcel_weight_kg": rng.choice(weights), "deadline_min": ddl, "feasible_stations": feasible})
                    break
    return customers
