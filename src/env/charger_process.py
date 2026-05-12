from __future__ import annotations


def occupy_charger(available_chargers: int, duration: float) -> int:
    if duration > 0 and available_chargers > 0:
        return available_chargers - 1
    return available_chargers


def release_charger(available_chargers: int, total_chargers: int) -> int:
    return min(total_chargers, available_chargers + 1)
