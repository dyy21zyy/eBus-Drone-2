from __future__ import annotations

import random
import math

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
    operation_cfg = config.get("operation", {})
    planned_headway_min = float(instance_cfg.get("planned_headway_min", config.get("planned_headway_min", 0.0)))
    if planned_headway_min <= 0:
        raise ValueError("planned_headway_min must be > 0 to construct scheduled trips and physical fleet size")
    nominal_line_time_min = float(operation_cfg.get("nominal_line_time_min", travel[0][-1] if n > 1 else 0.0))
    return_time_min = float(operation_cfg.get("return_time_min", nominal_line_time_min))
    layover_time_min = float(operation_cfg.get("layover_time_min", 0.0))
    nominal_cycle_time_min = nominal_line_time_min + return_time_min + layover_time_min
    physical_fleet_size = int(math.ceil(nominal_cycle_time_min / planned_headway_min))
    trips = []
    num_scheduled = int(instance_cfg.get("num_scheduled_trips", instance_cfg["num_scheduled_bus_trips"]))
    num_freight = int(instance_cfg.get("num_freight_carrying_trips", num_scheduled))
    for b in range(1, num_scheduled + 1):
        departure_min = (b - 1) * planned_headway_min
        terminal_arrival_min = departure_min + nominal_line_time_min
        return_completion_min = terminal_arrival_min + return_time_min
        layover_completion_min = return_completion_min + layover_time_min
        trips.append({
            "trip_id": b,
            "departure_min": departure_min,
            "terminal_arrival_min": terminal_arrival_min,
            "return_completion_min": return_completion_min,
            "layover_completion_min": layover_completion_min,
        })
    physical_buses = [f"v{idx+1}" for idx in range(max(1, physical_fleet_size))]
    next_available = {vid: 0.0 for vid in physical_buses}
    vehicle_circulation: dict[int, str] = {}
    for t in trips:
        tid = int(t["trip_id"])
        dep = float(t["departure_min"])
        feasible = [vid for vid in physical_buses if float(next_available[vid]) <= dep + 1e-9]
        if not feasible:
            raise ValueError(
                f"Infeasible vehicle circulation for trip {tid}: no physical bus available by departure "
                f"{dep:.3f} min. Increase fleet size or reduce cycle time/headway."
            )
        chosen = min(feasible, key=lambda vid: (next_available[vid], vid))
        vehicle_circulation[tid] = chosen
        next_available[chosen] = float(t["layover_completion_min"])
    freight_trip_ids = _select_freight_trip_ids(trips, num_freight, seed=seed)
    return {
        "stops": stops,
        "distances_km": distances,
        "nominal_travel_time_min": travel,
        "scheduled_bus_trips": trips,
        "freight_carrying_trip_ids": freight_trip_ids,
        "num_scheduled_trips": num_scheduled,
        "num_freight_carrying_trips": len(freight_trip_ids),
        "planned_headway_min": planned_headway_min,
        "physical_buses": physical_buses,
        "vehicle_circulation": vehicle_circulation,
        "return_time_min": return_time_min,
        "layover_time_min": layover_time_min,
        "nominal_cycle_time_min": nominal_cycle_time_min,
        "nominal_line_time_min": nominal_line_time_min,
        "physical_fleet_size": len(physical_buses),
    }
