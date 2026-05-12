from dataclasses import dataclass


@dataclass(frozen=True)
class BusTrip:
    trip_id: int
    departure_min: float
    passenger_capacity: int
    freight_capacity_kg: float
