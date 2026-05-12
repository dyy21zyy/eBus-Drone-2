from dataclasses import dataclass


@dataclass(frozen=True)
class ChargerSpec:
    power_kw: float
    efficiency: float
    action_set_seconds: list[int]
