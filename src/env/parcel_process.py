from __future__ import annotations


def unload_parcels(onboard_parcels: int, q_f: int) -> tuple[int, int]:
    unloaded = min(max(onboard_parcels, 0), max(q_f, 0))
    return onboard_parcels - unloaded, unloaded


def get_unloading_parcels(trip_id: int, station_id: int, assignment_index: dict, parcel_states: dict) -> list[int]:
    assigned = assignment_index.get("by_trip_station", {}).get((int(trip_id), int(station_id)), [])
    out: list[int] = []
    for pid in assigned:
        p = parcel_states.get(int(pid))
        if p is None:
            raise ValueError(f"Missing parcel state for parcel {pid}")
        if int(p["assigned_trip_id"]) != int(trip_id) or int(p["assigned_station_id"]) != int(station_id):
            raise ValueError(f"Parcel {pid} assignment mismatch for trip/station")
        if p["status"] == "delivered":
            raise ValueError(f"Parcel {pid} already delivered")
        if p["status"] == "onboard":
            out.append(int(pid))
        elif p["status"] in {"in_locker", "locker"}:
            raise ValueError(f"Parcel {pid} already unloaded")
    return out


def compute_unloading_volume(parcel_ids: list[int], parcel_states: dict) -> float:
    return float(sum(float(parcel_states[int(pid)]["weight_kg"]) for pid in parcel_ids))


def unload_parcels_to_locker(trip_state: dict, station_state: dict, parcel_states: dict, parcel_ids: list[int], release_time: float) -> list[int]:
    unloaded: list[int] = []
    for pid in parcel_ids:
        p = parcel_states[int(pid)]
        if p["status"] != "onboard":
            raise ValueError(f"Parcel {pid} cannot be unloaded from status {p['status']}")
        if int(pid) not in trip_state["onboard_parcel_ids"]:
            raise ValueError(f"Parcel {pid} not present onboard for trip {trip_state['trip_id']}")
        trip_state["onboard_parcel_ids"].remove(int(pid))
        p["status"] = "in_locker"
        p["release_time_min"] = float(release_time)
        p["release_time"] = float(release_time)
        p["station_id"] = int(station_state["station_id"])
        station_state.setdefault("locker_parcels", []).append(int(pid))
        unloaded.append(int(pid))

    trip_state["onboard_parcel_weight"] = float(sum(parcel_states[pid]["weight_kg"] for pid in trip_state["onboard_parcel_ids"]))
    station_state["locker_inventory_kg"] = float(sum(parcel_states[pid]["weight_kg"] for pid in station_state.get("locker_parcels", []) if parcel_states[pid]["status"] == "in_locker"))
    return unloaded
