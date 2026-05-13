from __future__ import annotations

from itertools import combinations, permutations

import numpy as np

from .drone_dispatch_model import parcel_priority_cost

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - optional dependency path
    linear_sum_assignment = None


def _build_cost_matrix(feasible_waiting: list[dict], now: float, eta_l_d: float, eta_u_d: float, n_rows: int) -> np.ndarray:
    parcel_costs = np.asarray(
        [parcel_priority_cost(parcel, now, eta_l_d, eta_u_d) for parcel in feasible_waiting],
        dtype=float,
    )
    # Drones are homogeneous in the paper model, so row costs are identical.
    return np.tile(parcel_costs, (n_rows, 1))


def _solve_exact_without_scipy(cost_matrix: np.ndarray, n_disp: int) -> list[tuple[int, int]]:
    n_drones, n_parcels = cost_matrix.shape
    best: tuple[float, tuple[int, ...], tuple[int, ...]] | None = None
    drone_ids = tuple(range(n_drones))
    parcel_ids = tuple(range(n_parcels))
    for drone_subset in combinations(drone_ids, n_disp):
        for parcel_subset in combinations(parcel_ids, n_disp):
            for perm in permutations(parcel_subset):
                total = float(sum(cost_matrix[drone_subset[i], perm[i]] for i in range(n_disp)))
                candidate = (total, drone_subset, perm)
                if best is None or candidate < best:
                    best = candidate
    if best is None:
        return []
    _, drones, parcels = best
    return [(int(drones[i]), int(parcels[i])) for i in range(n_disp)]


def solve_station_dispatch(
    idle_drone_ids: list[str],
    full_batteries: int,
    feasible_waiting: list[dict],
    now: float,
    eta_l_d: float = 1.0,
    eta_u_d: float = 1.0,
) -> tuple[list[dict], int]:
    n_disp = min(len(idle_drone_ids), int(full_batteries), len(feasible_waiting))
    if n_disp <= 0:
        return [], 0

    sorted_drones = sorted(idle_drone_ids)
    cost_matrix = _build_cost_matrix(feasible_waiting, now, eta_l_d, eta_u_d, len(sorted_drones))

    if linear_sum_assignment is not None:
        padded_cols = max(cost_matrix.shape[1], cost_matrix.shape[0])
        padded = np.full((cost_matrix.shape[0], padded_cols), fill_value=1e9, dtype=float)
        padded[:, : cost_matrix.shape[1]] = cost_matrix
        rows, cols = linear_sum_assignment(padded)
        valid = [(int(r), int(c)) for r, c in zip(rows.tolist(), cols.tolist()) if c < cost_matrix.shape[1]]
        valid.sort(key=lambda rc: (cost_matrix[rc[0], rc[1]], sorted_drones[rc[0]], int(feasible_waiting[rc[1]]["id"])))
        selected_pairs = valid[:n_disp]
    else:
        selected_pairs = _solve_exact_without_scipy(cost_matrix, n_disp)

    assignments = [
        {
            "drone_id": sorted_drones[drone_idx],
            "parcel_id": int(feasible_waiting[parcel_idx]["id"]),
            "parcel": feasible_waiting[parcel_idx],
        }
        for drone_idx, parcel_idx in selected_pairs
    ]
    return assignments, len(assignments)


def solve_greedy_dispatch(*args, **kwargs):
    """Backward-compatible solver API used by station operator."""
    return solve_station_dispatch(*args, **kwargs)
