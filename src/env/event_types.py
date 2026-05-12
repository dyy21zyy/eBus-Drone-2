from __future__ import annotations

from dataclasses import dataclass


@dataclass(order=True)
class Event:
    time: float
    trip_id: int
    stop_index: int
    station_id: int
    integrated: bool
    requires_stop: bool
    is_decision: bool
