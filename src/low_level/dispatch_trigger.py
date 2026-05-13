from __future__ import annotations


def should_trigger_dispatch(*, new_parcels: bool, drone_returned: bool, battery_fully_charged: bool, t: float, dispatch_interval: float) -> bool:
    if new_parcels or drone_returned or battery_fully_charged:
        return True
    if dispatch_interval <= 0:
        return False
    k = round(float(t) / float(dispatch_interval))
    return abs(float(t) - k * float(dispatch_interval)) <= 1e-9
