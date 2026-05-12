from __future__ import annotations


def idle_drone_ids(station_state: dict) -> list[str]:
    return [str(d.get("drone_id", d.get("id"))) for d in station_state.get("drones", []) if d.get("status") == "idle"]


def mark_active(station_state: dict, assignments: list[dict]) -> None:
    by_id = {str(a["drone_id"]): a for a in assignments}
    for d in station_state.get("drones", []):
        did = str(d.get("drone_id", d.get("id")))
        if did in by_id:
            a = by_id[did]
            d["status"] = "active"
            d["assigned_parcel_id"] = int(a["parcel_id"])
            d["return_time"] = float(a["drone_return_time"])


def process_returns(station_state: dict, now: float) -> list[str]:
    returned: list[str] = []
    for d in station_state.get("drones", []):
        if d.get("status") == "active" and d.get("return_time", float("inf")) <= now:
            d["status"] = "idle"
            d["return_time"] = None
            d["assigned_parcel_id"] = None
            returned.append(str(d.get("drone_id", d.get("id"))))
    return returned
