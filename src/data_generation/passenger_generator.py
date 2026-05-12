import random

def truncated_normal(rng: random.Random, mean: float, std: float, lo: float, hi: float) -> float:
    for _ in range(1000):
        x = rng.gauss(mean, std)
        if lo <= x <= hi:
            return x
    return min(hi, max(lo, mean))


def generate_passenger_parameters(num_stops: int, cfg: dict, rng: random.Random) -> dict:
    p = cfg["passenger"]
    lambdas = [
        truncated_normal(rng, p["baseline_lambda_mean"], p["baseline_lambda_std"], p["baseline_lambda_min"], p["baseline_lambda_max"])
        for _ in range(num_stops)
    ]
    alight = [rng.uniform(p["alighting_prob_min"], p["alighting_prob_max"]) for _ in range(num_stops)]
    return {"baseline_arrival_rate_per_min": lambdas, "alighting_probability": alight}
