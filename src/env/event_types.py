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
    passenger_service_preview: dict = field(compare=False, default_factory=dict)
    arrival_queue_before_preview: int = field(compare=False, default=0)
    arrival_onboard_before_preview: int = field(compare=False, default=0)
    unloading_volume_kg: float = field(compare=False, default=0.0)
