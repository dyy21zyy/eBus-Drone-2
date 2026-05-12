from dataclasses import dataclass

@dataclass(frozen=True)
class Event:
    time_min: int
    event_type: str
    payload: dict
