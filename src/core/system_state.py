from dataclasses import dataclass


@dataclass
class SystemState:
    time_min: float = 0.0
    onboard_passengers: int = 0
    onboard_parcel_weight_kg: float = 0.0
