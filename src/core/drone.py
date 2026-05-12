from dataclasses import dataclass


@dataclass(frozen=True)
class DroneSpec:
    speed_kmh: float
    max_round_trip_duration_min: float
    customer_service_time_min: float
    turnaround_time_min: float
