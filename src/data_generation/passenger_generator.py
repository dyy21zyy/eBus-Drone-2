from __future__ import annotations
import random


def _truncnorm(rng: random.Random, mean: float, std: float, lo: float, hi: float) -> float:
    while True:
        x = rng.gauss(mean, std)
        if lo <= x <= hi:
            return x


def generate_passenger_parameters(config: dict, stops: list[dict], seed: int) -> dict:
    rng = random.Random(seed + 31)
    p = config["passenger"]
    rates = {s["stop_id"]: _truncnorm(rng, p["baseline_rate_mean_per_min"], p["baseline_rate_std_per_min"], p["baseline_rate_min_per_min"], p["baseline_rate_max_per_min"]) for s in stops}
    return {"baseline_arrival_rate_per_stop_per_min": rates}


def generate_passenger_scenario(config: dict, stops: list[dict], seed: int) -> dict:
    rng = random.Random(seed + 41)
    horizon = int(config["generation"]["horizon_minutes"])
    base = generate_passenger_parameters(config, stops, seed)["baseline_arrival_rate_per_stop_per_min"]
    factor = float(config["passenger"]["demand_intensity_factor"])
    almin, almax = float(config["passenger"]["alighting_probability_min"]), float(config["passenger"]["alighting_probability_max"])
    rates = {}
    series = {}
    for s in stops:
        lam = max(base[s["stop_id"]] * factor, 0.0)
        rates[s["stop_id"]] = lam
        arrivals = [1 if rng.random() < min(lam, 1.0) else 0 for _ in range(horizon)]
        series[s["stop_id"]] = arrivals
    return {"passenger_arrivals": series, "arrival_rate_per_stop_per_min": rates, "alighting_probability": round(rng.uniform(almin, almax), 4)}
