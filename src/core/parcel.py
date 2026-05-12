from dataclasses import dataclass


@dataclass(frozen=True)
class Parcel:
    customer_id: int
    weight_kg: float
    deadline_min: float
