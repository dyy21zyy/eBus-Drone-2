from __future__ import annotations

from .battery_charging import charging_load_kw, complete_charging_jobs, start_charging_jobs


def consume_full_batteries(station_state: dict, n: int) -> None:
    station_state["full_batteries"] = max(0, int(station_state.get("full_batteries", 0)) - int(n))


def add_depleted_batteries(station_state: dict, n: int) -> None:
    cur = int(station_state.get("depleted_batteries", station_state.get("empty_batteries", 0)))
    station_state["depleted_batteries"] = cur + int(n)
    station_state["empty_batteries"] = station_state["depleted_batteries"]


def charge_depleted_batteries(station_state: dict, now: float, p_e: float, p_l: float) -> dict:
    completed = complete_charging_jobs(station_state, now)
    g = start_charging_jobs(station_state, now, p_e, p_l)
    return {"g": g, "completed": completed, "P_D": charging_load_kw(station_state)}
