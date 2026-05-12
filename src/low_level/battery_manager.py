from __future__ import annotations


def consume_full_batteries(station_state: dict, n: int) -> None:
    station_state["full_batteries"] = max(0, int(station_state.get("full_batteries", 0)) - int(n))


def add_depleted_batteries(station_state: dict, n: int) -> None:
    station_state["empty_batteries"] = int(station_state.get("empty_batteries", 0)) + int(n)


def charge_depleted_batteries(station_state: dict, p_e: float, p_l: float) -> dict:
    b_empty = int(station_state.get("empty_batteries", 0))
    g_max = int(station_state.get("G_max", 0))
    p_capacity = float(station_state.get("P_capacity", 0.0))
    p_bat = float(station_state.get("P_bat", 1.0))
    residual_slots = max(0, int((p_capacity - p_e - p_l) // p_bat))
    g = min(b_empty, g_max, residual_slots)
    station_state["empty_batteries"] = b_empty - g
    station_state["full_batteries"] = int(station_state.get("full_batteries", 0)) + g
    return {"g": g, "P_D": g * p_bat}
