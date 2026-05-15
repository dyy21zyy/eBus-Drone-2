from __future__ import annotations

import math


def complete_charging_jobs(station_state: dict, now: float) -> int:
    jobs = station_state.setdefault("charging_batteries", station_state.setdefault("batteries_charging", []))
    completed = [j for j in jobs if float(j["completion_time_min"]) <= now]
    if completed:
        station_state["full_batteries"] = int(station_state.get("full_batteries", 0)) + len(completed)
    station_state["charging_batteries"] = [j for j in jobs if float(j["completion_time_min"]) > now]
    station_state["batteries_charging"] = station_state["charging_batteries"]
    return len(completed)


def start_charging_jobs(station_state: dict, now: float, p_e: float, p_l: float) -> int:
    jobs = station_state.setdefault("charging_batteries", station_state.setdefault("batteries_charging", []))
    depleted = int(station_state.get("depleted_batteries", station_state.get("empty_batteries", 0)))
    g_max = int(station_state.get("G_max", station_state.get("charging_slots", 0)))
    p_capacity = float(station_state.get("P_capacity", 0.0))
    p_bat = float(station_state.get("P_bat", station_state.get("battery_charge_power_kw", 1.0)))
    charge_duration = station_state.get("battery_charge_duration_min")
    if charge_duration is None:
        cap_kwh = station_state.get("battery_capacity_kwh")
        if cap_kwh is not None and p_bat > 0:
            charge_duration = 60.0 * float(cap_kwh) / float(p_bat)
        else:
            charge_duration = 45.0
    charge_duration = float(charge_duration)

    active_jobs = len(jobs)
    available_slots = max(0, g_max - active_jobs)
    residual_power_after_active_jobs = p_capacity - float(p_e) - float(p_l) - active_jobs * p_bat
    available_power_slots = max(0, math.floor(residual_power_after_active_jobs / p_bat)) if p_bat > 0 else 0
    g = min(depleted, available_slots, available_power_slots)
    for _ in range(max(0, g)):
        jobs.append({"start_time_min": now, "completion_time_min": now + charge_duration})
    station_state["depleted_batteries"] = depleted - max(0, g)
    station_state["empty_batteries"] = station_state["depleted_batteries"]
    station_state["batteries_charging"] = jobs
    return max(0, g)


def charging_load_kw(station_state: dict) -> float:
    p_bat = float(station_state.get("P_bat", station_state.get("battery_charge_power_kw", 1.0)))
    jobs = station_state.get("charging_batteries", station_state.get("batteries_charging", []))
    return len(jobs) * p_bat
