from __future__ import annotations

from collections import defaultdict

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


def _greedy_fallback(data):
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

    result = AssignmentResult(decisions=decisions, total_cost=sum(d.planned_cost for d in decisions), method="greedy_fallback")
    validate_assignment(data, result)
    return result


def solve_assignment(data):
    model = build_assignment_model(data)
    idx = {k: i for i, k in enumerate(model.variable_keys)}
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
        if not res.success:
            raise AssignmentInfeasibleError(f"MILP failed: {res.message}")

        x = res.x
        decisions = []
        for key, j in idx.items():
            if x[j] >= 0.5:
                b, h, i = key
                decisions.append(AssignmentDecision(customer_id=i, trip_id=b, station_id=h, planned_cost=_objective_for_key(data, key)))
        result = AssignmentResult(decisions=decisions, total_cost=float(sum(d.planned_cost for d in decisions)), method="scipy_milp")
        validate_assignment(data, result)
        return result
    except Exception:
        return _greedy_fallback(data)
