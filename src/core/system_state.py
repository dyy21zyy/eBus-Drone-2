from dataclasses import dataclass

@dataclass
class SystemState:
    time_min: int
    waiting_passengers: list[int]
    station_power_kw: dict[int, float]
