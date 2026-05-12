from __future__ import annotations


def apply_charge(battery: float, duration: float, rate: float, battery_max: float) -> float:
    return min(battery_max, battery + max(duration, 0.0) * rate)


def apply_travel_consumption(battery: float, consumption: float) -> float:
    return battery - max(consumption, 0.0)
