from dataclasses import dataclass

@dataclass(frozen=True)
class ParcelOrder:
    customer_id: int
    weight_kg: float
    deadline_min: int
