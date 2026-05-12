from dataclasses import dataclass

@dataclass(frozen=True)
class IntegratedStation:
    stop_id: int
    chargers: int
    drones: int
    locker_capacity_kg: float
