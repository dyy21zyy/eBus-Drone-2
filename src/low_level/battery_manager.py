from __future__ import annotations


def consume_full_batteries(station_state: dict, n: int) -> None:
    station_state["full_batteries"] = max(0, int(station_state.get("full_batteries", 0)) - int(n))


def add_depleted_batteries(station_state: dict, n: int) -> None:
    cur = int(station_state.get("depleted_batteries", station_state.get("empty_batteries", 0)))
    station_state["depleted_batteries"] = cur + int(n)
    station_state["empty_batteries"] = station_state["depleted_batteries"]


def charge_depleted_batteries(station_state: dict, now: float, p_e: float, p_l: float) -> dict:
    charging = station_state.setdefault("charging_batteries", [])
    completed = [t for t in charging if float(t) <= now]
    if completed:
        station_state["full_batteries"] = int(station_state.get("full_batteries", 0)) + len(completed)
    station_state["charging_batteries"] = [t for t in charging if float(t) > now]
    b_empty = int(station_state.get("depleted_batteries", station_state.get("empty_batteries", 0)))
    g_max = int(station_state.get("G_max", station_state.get("charging_slots", 0)))
    p_capacity = float(station_state.get("P_capacity", 0.0))
    p_bat = float(station_state.get("P_bat", 1.0))
    charge_duration = float(station_state.get("battery_charge_duration_min", 10.0))
    residual_slots = max(0, int((p_capacity - p_e - p_l) // p_bat))
    available_slots = max(0, g_max - len(station_state["charging_batteries"]))
    g = min(b_empty, available_slots, residual_slots)
    for _ in range(g):
        station_state["charging_batteries"].append(now + charge_duration)
    station_state["depleted_batteries"] = b_empty - g
    station_state["empty_batteries"] = station_state["depleted_batteries"]
    return {"g": g, "completed": len(completed), "P_D": (len(station_state["charging_batteries"]) * p_bat)}
