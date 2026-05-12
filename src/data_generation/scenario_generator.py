from src.data_generation.passenger_generator import generate_passenger_parameters
from src.data_generation.power_load_generator import generate_station_base_load


def generate_scenario(instance: dict, cfg: dict, seed: int) -> dict:
    import random
    rng = random.Random(seed)
    num_stops = instance["network"]["num_stops"]
    passenger_params = generate_passenger_parameters(num_stops, cfg, rng)
    station_load = generate_station_base_load(instance["stations"], cfg["network"]["horizon_minutes"], cfg, rng)
    arrivals = []
    for lam in passenger_params["baseline_arrival_rate_per_min"]:
        stop_series = [1 if rng.random() < min(0.95, lam * cfg["passenger"]["intensity_factor"]) else 0 for _ in range(cfg["network"]["horizon_minutes"])]
        arrivals.append(stop_series)
    return {
        "seed": seed,
        "passenger_arrivals": arrivals,
        "alighting_probability": passenger_params["alighting_probability"],
        "station_base_load": station_load,
        "disturbances": {"bus_delay_min": [rng.randint(0, 2) for _ in instance["bus_trips"]]},
    }
