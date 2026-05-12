
def generate_corridor_network(num_stops: int, stop_spacing_km: float, horizon_minutes: int) -> dict:
    stops = [{"stop_id": i + 1, "position_km": i * stop_spacing_km} for i in range(num_stops)]
    distances = [[abs(i - j) * stop_spacing_km for j in range(num_stops)] for i in range(num_stops)]
    nominal_speed_kmh = 30.0
    travel_times_min = [[d / nominal_speed_kmh * 60.0 for d in row] for row in distances]
    return {
        "num_stops": num_stops,
        "horizon_minutes": horizon_minutes,
        "stops": stops,
        "distances_km": distances,
        "nominal_travel_times_min": travel_times_min,
    }
