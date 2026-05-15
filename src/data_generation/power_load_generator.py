from __future__ import annotations
import random


def generate_station_base_load(config: dict, station_ids: list[int], seed: int) -> dict:
    rng = random.Random(seed + 53)
    gen = config.get("generation", {})
    horizon = int(gen.get("delivery_evaluation_horizon_minutes", gen.get("horizon_minutes", 480)))
    lo, hi = float(config["power"]["nominal_base_load_min_kw"]), float(config["power"]["nominal_base_load_max_kw"])
    std = float(config["power"]["disturbance_std_kw"])
    loads = {}
    for sid in station_ids:
        nominal = rng.uniform(lo, hi)
        loads[sid] = [round(max(0.0, nominal + rng.gauss(0.0, std)), 3) for _ in range(horizon)]
    return {"station_base_load_kw": loads}
