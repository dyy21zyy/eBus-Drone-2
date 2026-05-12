from __future__ import annotations

from .drone_dispatch_model import parcel_priority_cost


def solve_greedy_dispatch(idle_drone_ids: list[str], full_batteries: int, feasible_waiting: list[dict], now: float, eta_l_d: float = 1.0, eta_u_d: float = 1.0) -> tuple[list[dict], int]:
    n_disp = min(len(idle_drone_ids), int(full_batteries), len(feasible_waiting))
    if n_disp <= 0:
        return [], 0
    ranked = sorted(
        feasible_waiting,
        key=lambda p: (parcel_priority_cost(p, now, eta_l_d, eta_u_d), str(p["id"])),
    )
    chosen = ranked[:n_disp]
    assignments = []
    for drone_id, parcel in zip(idle_drone_ids[:n_disp], chosen):
        assignments.append({"drone_id": drone_id, "parcel_id": int(parcel["id"]), "parcel": parcel})
    return assignments, n_disp
