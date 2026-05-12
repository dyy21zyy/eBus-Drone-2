from dataclasses import dataclass

@dataclass(frozen=True)
class DroneSpec:
    speed_kmh: float
    max_round_trip_mission_min: float
    service_time_min: float
    turnaround_time_min: float
