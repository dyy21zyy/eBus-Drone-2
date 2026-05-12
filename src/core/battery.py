from dataclasses import dataclass

@dataclass(frozen=True)
class BatteryPool:
    charged: int
    depleted: int
    charging_power_kw: float
    max_simultaneous_charging: int
