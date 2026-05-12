from dataclasses import dataclass

@dataclass(frozen=True)
class ChargerSpec:
    pantograph_power_kw: float
    efficiency: float
    max_single_stop_seconds: int
    action_set_seconds: list[int]
