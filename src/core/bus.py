from dataclasses import dataclass

@dataclass(frozen=True)
class ScheduledBusTrip:
    trip_id: int
    departure_min: int
    passenger_capacity: int
    freight_capacity_kg: float
