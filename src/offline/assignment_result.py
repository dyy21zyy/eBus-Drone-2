from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class AssignmentDecision:
    customer_id: int
    trip_id: int
    station_id: int
    planned_cost: float


@dataclass
class AssignmentResult:
    decisions: list[AssignmentDecision]
    total_cost: float
    method: str

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "total_cost": self.total_cost,
            "decisions": [asdict(d) for d in self.decisions],
        }

    @staticmethod
    def from_dict(payload: dict) -> "AssignmentResult":
        return AssignmentResult(
            decisions=[AssignmentDecision(**d) for d in payload.get("decisions", [])],
            total_cost=float(payload.get("total_cost", 0.0)),
            method=str(payload.get("method", "unknown")),
        )
