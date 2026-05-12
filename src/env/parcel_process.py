from __future__ import annotations


def unload_parcels(onboard_parcels: int, q_f: int) -> tuple[int, int]:
    unloaded = min(max(onboard_parcels, 0), max(q_f, 0))
    return onboard_parcels - unloaded, unloaded
