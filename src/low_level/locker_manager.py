from __future__ import annotations


def feasible_parcels(waiting_parcels: list[dict], now: float) -> list[dict]:
    """Filter waiting parcels by station-feasibility constraints."""
    out: list[dict] = []
    for p in waiting_parcels:
        if p.get("feasible", True):
            out.append(p)
    return out


def remove_parcels_by_id(waiting_parcels: list[dict], parcel_ids: list[str]) -> list[dict]:
    ids = set(parcel_ids)
    return [p for p in waiting_parcels if str(p["id"]) not in ids]
