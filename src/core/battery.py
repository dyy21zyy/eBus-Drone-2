from dataclasses import dataclass


@dataclass(frozen=True)
class BatteryPool:
    fully_charged: int
    depleted: int
    max_simultaneous_charging: int
