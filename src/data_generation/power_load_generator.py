import random


def generate_station_base_load(station_ids: list[int], horizon_minutes: int, cfg: dict, rng: random.Random) -> dict:
    lo = cfg["station_power"]["nominal_base_load_min_kw"]
    hi = cfg["station_power"]["nominal_base_load_max_kw"]
    data = {}
    for sid in station_ids:
        nominal = rng.uniform(lo, hi)
        series = [nominal + rng.uniform(-10.0, 10.0) for _ in range(horizon_minutes)]
        data[sid] = {"nominal_kw": nominal, "series_kw": series}
    return data
