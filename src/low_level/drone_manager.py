from __future__ import annotations


def idle_drone_ids(station_state: dict) -> list[str]:
    return [d["id"] for d in station_state.get("drones", []) if d.get("status") == "idle"]


def mark_active(station_state: dict, drone_ids: list[str], return_time: float) -> None:
    dids = set(drone_ids)
    for d in station_state.get("drones", []):
        if d["id"] in dids:
            d["status"] = "active"
            d["return_time"] = return_time


def process_returns(station_state: dict, now: float) -> int:
    returned = 0
    for d in station_state.get("drones", []):
        if d.get("status") == "active" and d.get("return_time", float("inf")) <= now:
            d["status"] = "idle"
            d["return_time"] = None
            returned += 1
    return returned
