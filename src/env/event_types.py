from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(order=True)
class Event:
    time: float
    kind: str = field(compare=True, default="bus_arrival")
    trip_id: int = field(compare=False, default=-1)
    stop_index: int = field(compare=False, default=-1)
    station_id: int = field(compare=False, default=-1)
    integrated: bool = field(compare=False, default=False)
    requires_stop: bool = field(compare=False, default=False)
    is_decision: bool = field(compare=False, default=False)
    passengers_required: bool = field(compare=False, default=False)
    parcel_required: bool = field(compare=False, default=False)
