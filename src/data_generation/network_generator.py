from __future__ import annotations

import random

def _select_freight_trip_ids(trips: list[dict], num_freight: int, seed: int) -> list[int]:
    if num_freight <= 0:
        return []
    if num_freight >= len(trips):
        return [int(t["trip_id"]) for t in trips]
    if num_freight == 1:
        return [int(trips[0]["trip_id"])]
    max_idx = len(trips) - 1
    step = max_idx / float(num_freight - 1)
    candidate_idxs = [int(round(j * step)) for j in range(num_freight)]
    rng = random.Random(int(seed))
    selected_idxs = set()
    for idx in candidate_idxs:
        if idx not in selected_idxs:
            selected_idxs.add(idx)
            continue
        for alt in sorted(range(len(trips)), key=lambda x: (abs(x - idx), rng.random())):
            if alt not in selected_idxs:
                selected_idxs.add(alt)
                break
    return [int(trips[idx]["trip_id"]) for idx in sorted(selected_idxs)]


def generate_network(config: dict, instance_cfg: dict, seed: int) -> dict:
    n = int(config["network"]["num_stops"])
    spacing = float(config["network"]["stop_spacing_km"])
    speed = float(config["bus"]["nominal_speed_kmh"])
    stops = [{"stop_id": i, "x_km": (i - 1) * spacing, "y_km": 0.0} for i in range(1, n + 1)]
    distances = [[abs(i - j) * spacing for j in range(n)] for i in range(n)]
    travel = [[(d / speed) * 60.0 for d in row] for row in distances]
    trips = []
    num_scheduled = int(instance_cfg.get("num_scheduled_trips", instance_cfg["num_scheduled_bus_trips"]))
    num_freight = int(instance_cfg.get("num_freight_carrying_trips", num_scheduled))
    for b in range(1, num_scheduled + 1):
        trips.append({"trip_id": b, "departure_min": (b - 1) * float(instance_cfg["planned_headway_min"])})
    freight_trip_ids = _select_freight_trip_ids(trips, num_freight, seed=seed)
    return {
        "stops": stops,
        "distances_km": distances,
        "nominal_travel_time_min": travel,
        "scheduled_bus_trips": trips,
        "freight_carrying_trip_ids": freight_trip_ids,
        "num_scheduled_trips": num_scheduled,
        "num_freight_carrying_trips": len(freight_trip_ids),
    }
