from __future__ import annotations


def should_trigger_dispatch(*, new_parcels: bool, drone_returned: bool, battery_fully_charged: bool, t: float, dispatch_interval: float) -> bool:
    if new_parcels or drone_returned or battery_fully_charged:
        return True
    return dispatch_interval > 0 and t % dispatch_interval == 0
