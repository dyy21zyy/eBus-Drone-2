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
    solver_name: str = "unknown"
    status: str = "unknown"
    objective_value: float | None = None
    runtime_sec: float | None = None
    mip_gap: float | None = None
    used_fallback: bool = False
    fallback_reason: str | None = None
    number_assigned_customers: int = 0
    number_customers: int = 0
    feasibility_summary: dict | None = None
    cost_components: dict | None = None

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "total_cost": self.total_cost,
            "solver_name": self.solver_name,
            "status": self.status,
            "objective_value": self.objective_value,
            "runtime_sec": self.runtime_sec,
            "mip_gap": self.mip_gap,
            "used_fallback": self.used_fallback,
            "fallback_reason": self.fallback_reason,
            "number_assigned_customers": self.number_assigned_customers,
            "number_customers": self.number_customers,
            "feasibility_summary": self.feasibility_summary,
            "cost_components": self.cost_components,
            "decisions": [asdict(d) for d in self.decisions],
        }

    @staticmethod
    def from_dict(payload: dict) -> "AssignmentResult":
        return AssignmentResult(
            decisions=[AssignmentDecision(**d) for d in payload.get("decisions", [])],
            total_cost=float(payload.get("total_cost", 0.0)),
            method=str(payload.get("method", "unknown")),
            solver_name=str(payload.get("solver_name", "unknown")),
            status=str(payload.get("status", "unknown")),
            objective_value=payload.get("objective_value"),
            runtime_sec=payload.get("runtime_sec"),
            mip_gap=payload.get("mip_gap"),
            used_fallback=bool(payload.get("used_fallback", False)),
            fallback_reason=payload.get("fallback_reason"),
            number_assigned_customers=int(payload.get("number_assigned_customers", 0)),
            number_customers=int(payload.get("number_customers", 0)),
            feasibility_summary=payload.get("feasibility_summary"),
            cost_components=payload.get("cost_components"),
        )
