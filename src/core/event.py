from dataclasses import dataclass


@dataclass(order=True, frozen=True)
class Event:
    time_min: float
    event_type: str
    payload: dict
