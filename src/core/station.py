from dataclasses import dataclass


@dataclass(frozen=True)
class Station:
    station_id: int
    chargers: int
    locker_capacity_kg: float
    station_power_capacity_kw: float
