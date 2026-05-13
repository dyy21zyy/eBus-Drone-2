from __future__ import annotations

from collections import defaultdict
import time

from src.offline.assignment_model import build_assignment_model
from src.offline.assignment_result import AssignmentDecision, AssignmentResult
from src.offline.assignment_validator import validate_assignment


class AssignmentInfeasibleError(ValueError):
    pass


def _objective_for_key(data, key):
    b, h, i = key
    return (
        data.c_b[(b, h)] * data.parcel_weight[i]
        + data.c_d[(h, i)]
        + data.beta_h * data.parcel_weight[i] * data.h_bhi_0[key]
        + data.beta_l * data.lateness_0_plus[key]
    )


def _cost_components(data, decisions):
    bus = sum(data.c_b[(d.trip_id, d.station_id)] * data.parcel_weight[d.customer_id] for d in decisions)
    drone = sum(data.c_d[(d.station_id, d.customer_id)] for d in decisions)
    hold = sum(data.beta_h * data.parcel_weight[d.customer_id] * data.h_bhi_0[(d.trip_id, d.station_id, d.customer_id)] for d in decisions)
    late = sum(data.beta_l * data.lateness_0_plus[(d.trip_id, d.station_id, d.customer_id)] for d in decisions)
    return {
        "bus_transport_cost": float(bus),
        "drone_delivery_cost": float(drone),
        "planned_locker_holding_cost": float(hold),
        "planned_lateness_cost": float(late),
        "total_objective": float(bus + drone + hold + late),
    }


def _greedy_fallback(data, fallback_reason: str):
    bus_load = defaultdict(float)
    unload_load = defaultdict(float)
    drone_work = defaultdict(float)
    locker_load = {(h, tau): 0.0 for h in data.stations for tau in data.t_grid}
    decisions = []

    for i in sorted(data.customers):
        candidates = []
        for h in sorted(data.feasible_stations_by_customer[i]):
            for b in data.trips:
                if data.delta_bh[(b, h)] != 1:
                    continue
                key = (b, h, i)
                candidates.append(( _objective_for_key(data, key), b, h, key))
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))

        chosen = None
        fail_reasons = set()
        for cost, b, h, key in candidates:
            q = data.parcel_weight[i]
            if bus_load[b] + q > data.q_f[b] + 1e-9:
                fail_reasons.add(f"bus freight capacity trip={b}")
                continue
            if unload_load[(b, h)] + q > data.q_u[h] + 1e-9:
                fail_reasons.add(f"unloading capacity trip={b},station={h}")
                continue
            if drone_work[h] + data.t_hi_rt[(h, i)] > data.num_drones[h] * data.operating_horizon + 1e-9:
                fail_reasons.add(f"drone workload station={h}")
                continue
            r = data.r_bhi_0[key]
            p = data.p_bhi_0[key]
            locker_ok = True
            for tau in data.t_grid:
                if r <= tau < p and locker_load[(h, tau)] + q > data.k[h] + 1e-9:
                    locker_ok = False
                    fail_reasons.add(f"locker capacity station={h},tau={tau}")
                    break
            if not locker_ok:
                continue
            chosen = (cost, b, h, key)
            break

        if chosen is None:
            raise AssignmentInfeasibleError(f"No feasible assignment for customer {i}; violated constraints: {sorted(fail_reasons)}")

        cost, b, h, key = chosen
        q = data.parcel_weight[i]
        bus_load[b] += q
        unload_load[(b, h)] += q
        drone_work[h] += data.t_hi_rt[(h, i)]
        for tau in data.t_grid:
            if data.r_bhi_0[key] <= tau < data.p_bhi_0[key]:
                locker_load[(h, tau)] += q
        decisions.append(AssignmentDecision(customer_id=i, trip_id=b, station_id=h, planned_cost=cost))

    components = _cost_components(data, decisions)
    result = AssignmentResult(
        decisions=decisions, total_cost=components["total_objective"], method="greedy_fallback",
        solver_name="greedy_fallback", status="fallback_feasible", objective_value=components["total_objective"],
        used_fallback=True, fallback_reason=fallback_reason, number_assigned_customers=len(decisions),
        number_customers=len(data.customers), feasibility_summary={"all_customers_assigned_exactly_once": True},
        cost_components=components
    )
    validate_assignment(data, result)
    return result


def solve_assignment(data, allow_greedy_fallback: bool = False):
    model = build_assignment_model(data)
    idx = {k: i for i, k in enumerate(model.variable_keys)}
    start = time.perf_counter()
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp

        n = len(model.variable_keys)
        c = np.array(model.objective, dtype=float)
        integrality = np.ones(n, dtype=int)
        bounds = Bounds(np.zeros(n), np.ones(n))
        constraints = []

        for i in data.customers:
            row = np.zeros(n)
            for h in data.feasible_stations_by_customer[i]:
                for b in data.trips:
                    row[idx[(b, h, i)]] = 1.0
            constraints.append(LinearConstraint(row, 1.0, 1.0))

        for b in data.trips:
            row = np.zeros(n)
            for h in data.stations:
                for i in data.feasible_customers_by_station[h]:
                    row[idx[(b, h, i)]] = data.parcel_weight[i]
            constraints.append(LinearConstraint(row, -np.inf, data.q_f[b]))

        for b in data.trips:
            for h in data.stations:
                row = np.zeros(n)
                for i in data.feasible_customers_by_station[h]:
                    row[idx[(b, h, i)]] = data.parcel_weight[i]
                constraints.append(LinearConstraint(row, -np.inf, data.q_u[h]))

        for b, h, i in model.variable_keys:
            if data.delta_bh[(b, h)] == 0:
                row = np.zeros(n)
                row[idx[(b, h, i)]] = 1.0
                constraints.append(LinearConstraint(row, -np.inf, 0.0))

        for h in data.stations:
            for tau in data.t_grid:
                row = np.zeros(n)
                for b in data.trips:
                    for i in data.feasible_customers_by_station[h]:
                        key = (b, h, i)
                        if data.r_bhi_0[key] <= tau < data.p_bhi_0[key]:
                            row[idx[key]] = data.parcel_weight[i]
                constraints.append(LinearConstraint(row, -np.inf, data.k[h]))

        for h in data.stations:
            row = np.zeros(n)
            for b in data.trips:
                for i in data.feasible_customers_by_station[h]:
                    row[idx[(b, h, i)]] = data.t_hi_rt[(h, i)]
            constraints.append(LinearConstraint(row, -np.inf, data.num_drones[h] * data.operating_horizon))

        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        status = str(getattr(res, "message", "unknown"))
        if not res.success:
            raise AssignmentInfeasibleError(f"MILP failed: {status}")

        x = res.x
        decisions = []
        for key, j in idx.items():
            if x[j] >= 0.5:
                b, h, i = key
                decisions.append(AssignmentDecision(customer_id=i, trip_id=b, station_id=h, planned_cost=_objective_for_key(data, key)))
        components = _cost_components(data, decisions)
        result = AssignmentResult(
            decisions=decisions, total_cost=components["total_objective"], method="scipy_milp",
            solver_name="scipy.optimize.milp", status=status, objective_value=float(res.fun),
            runtime_sec=float(time.perf_counter() - start), mip_gap=getattr(res, "mip_gap", None),
            used_fallback=False, number_assigned_customers=len(decisions), number_customers=len(data.customers),
            feasibility_summary={"all_customers_assigned_exactly_once": len(decisions) == len(data.customers)},
            cost_components=components
        )
        validate_assignment(data, result)
        return result
    except Exception as exc:
        if not allow_greedy_fallback:
            raise AssignmentInfeasibleError(f"MILP solve failed and greedy fallback is disabled: {exc}") from exc
        return _greedy_fallback(data, fallback_reason=str(exc))
