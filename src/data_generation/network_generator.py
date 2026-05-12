from __future__ import annotations


def generate_network(config: dict, instance_cfg: dict) -> dict:
    n = int(config["network"]["num_stops"])
    spacing = float(config["network"]["stop_spacing_km"])
    speed = float(config["bus"]["nominal_speed_kmh"])
    stops = [{"stop_id": i, "x_km": (i - 1) * spacing, "y_km": 0.0} for i in range(1, n + 1)]
    distances = [[abs(i - j) * spacing for j in range(n)] for i in range(n)]
    travel = [[(d / speed) * 60.0 for d in row] for row in distances]
    trips = []
    for b in range(1, int(instance_cfg["num_scheduled_bus_trips"]) + 1):
        trips.append({"trip_id": b, "departure_min": (b - 1) * float(instance_cfg["planned_headway_min"])})
    return {"stops": stops, "distances_km": distances, "nominal_travel_time_min": travel, "scheduled_bus_trips": trips}
