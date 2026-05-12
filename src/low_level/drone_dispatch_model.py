from __future__ import annotations


def parcel_priority_cost(parcel: dict, now: float, eta_l_d: float, eta_u_d: float) -> float:
    t_out = float(parcel["T_out"])
    deadline = float(parcel["deadline"])
    base = float(parcel.get("c_D", 0.0))
    urgency = max(0.0, float(parcel.get("urgency", 0.0)))
    predicted_completion = now + t_out
    predicted_lateness = max(0.0, predicted_completion - deadline)
    return base + eta_l_d * predicted_lateness - eta_u_d * urgency
