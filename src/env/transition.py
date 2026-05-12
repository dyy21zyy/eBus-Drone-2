from __future__ import annotations


def advance_time(current_time: float, dwell: float, travel_time: float) -> float:
    return current_time + max(dwell, 0.0) + max(travel_time, 0.0)
