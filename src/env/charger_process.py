from __future__ import annotations


def occupy_charger(charger_release_times_min: list[float], now_min: float, duration_sec: float) -> bool:
    if duration_sec <= 0:
        return False
    for i, release_time in enumerate(charger_release_times_min):
        if float(release_time) <= float(now_min):
            charger_release_times_min[i] = float(now_min) + float(duration_sec) / 60.0
            return True
    return False


def available_chargers(charger_release_times_min: list[float], now_min: float) -> int:
    return sum(1 for t in charger_release_times_min if float(t) <= float(now_min))
