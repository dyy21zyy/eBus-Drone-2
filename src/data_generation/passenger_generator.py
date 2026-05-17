from __future__ import annotations
import random
import math


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
    gen = config.get("generation", {})
    horizon = int(gen.get("bus_operation_horizon_minutes", gen.get("horizon_minutes", 480)))
    base = generate_passenger_parameters(config, stops, seed)["baseline_arrival_rate_per_stop_per_min"]
    factor = float(config["passenger"]["demand_intensity_factor"])
    almin, almax = float(config["passenger"]["alighting_probability_min"]), float(config["passenger"]["alighting_probability_max"])
    rates, arrival_profiles, alighting_profiles, series = {}, {}, {}, {}
    for s in stops:
        sid = s["stop_id"]
        lam_base = max(base[sid] * factor, 0.0)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        lam_profile, mu_profile, arrivals = [], [], []
        for t in range(horizon):
            seasonal = 1.0 + 0.35 * math.sin((2.0 * math.pi * t / max(horizon, 1)) + phase)
            lam_t = max(0.0, lam_base * seasonal)
            lam_profile.append(round(lam_t, 6))
            arrivals.append(int(rng.random() < min(lam_t, 1.0)))
            mu_profile.append(round(rng.uniform(almin, almax), 6))
        rates[sid] = lam_base
        arrival_profiles[str(sid)] = lam_profile
        alighting_profiles[str(sid)] = mu_profile
        series[sid] = arrivals
    return {
        "passenger_arrivals": series,
        "arrival_rate_per_stop_per_min": rates,
        "arrival_rate_profile_per_stop_per_min": arrival_profiles,
        "alighting_probability": round(rng.uniform(almin, almax), 4),
        "alighting_profile_per_stop": alighting_profiles,
    }
