from __future__ import annotations

from collections import defaultdict
import time
import warnings

from src.offline.assignment_model import build_assignment_model
from src.offline.assignment_result import AssignmentDecision, AssignmentResult
from src.offline.assignment_validator import validate_assignment


class AssignmentInfeasibleError(ValueError):
    pass


class SolverUnavailableError(AssignmentInfeasibleError):
    pass


def _log(enabled: bool, msg: str):
    if enabled:
        print(msg, flush=True)


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


def _build_constraints(data, idx, n):
    import numpy as np
    from scipy.optimize import LinearConstraint
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
    for b, h, i in idx.keys():
        if data.delta_bh[(b, h)] == 0:
            row = np.zeros(n); row[idx[(b, h, i)]] = 1.0
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
    return constraints


def _extract_result(data, idx, x):
    decisions = []
    for key, j in idx.items():
        if x[j] >= 0.5:
            b, h, i = key
            decisions.append(AssignmentDecision(customer_id=i, trip_id=b, station_id=h, planned_cost=_objective_for_key(data, key)))
    return decisions


def _greedy_fallback(data, fallback_reason: str):
    # unchanged
    bus_load = defaultdict(float); unload_load = defaultdict(float); drone_work = defaultdict(float)
    locker_load = {(h, tau): 0.0 for h in data.stations for tau in data.t_grid}; decisions = []
    for i in sorted(data.customers):
        candidates = []
        for h in sorted(data.feasible_stations_by_customer[i]):
            for b in data.trips:
                if data.delta_bh[(b, h)] != 1: continue
                key = (b, h, i); candidates.append((_objective_for_key(data, key), b, h, key))
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        chosen = None; fail_reasons = set()
        for cost, b, h, key in candidates:
            q = data.parcel_weight[i]
            if bus_load[b] + q > data.q_f[b] + 1e-9: fail_reasons.add(f"bus freight capacity trip={b}"); continue
            if unload_load[(b, h)] + q > data.q_u[h] + 1e-9: fail_reasons.add(f"unloading capacity trip={b},station={h}"); continue
            if drone_work[h] + data.t_hi_rt[(h, i)] > data.num_drones[h] * data.operating_horizon + 1e-9: fail_reasons.add(f"drone workload station={h}"); continue
            r = data.r_bhi_0[key]; p = data.p_bhi_0[key]
            locker_ok = True
            for tau in data.t_grid:
                if r <= tau < p and locker_load[(h, tau)] + q > data.k[h] + 1e-9:
                    locker_ok = False; fail_reasons.add(f"locker capacity station={h},tau={tau}"); break
            if not locker_ok: continue
            chosen = (cost, b, h, key); break
        if chosen is None: raise AssignmentInfeasibleError(f"No feasible assignment for customer {i}; violated constraints: {sorted(fail_reasons)}")
        cost, b, h, key = chosen; q = data.parcel_weight[i]
        bus_load[b] += q; unload_load[(b, h)] += q; drone_work[h] += data.t_hi_rt[(h, i)]
        for tau in data.t_grid:
            if data.r_bhi_0[key] <= tau < data.p_bhi_0[key]: locker_load[(h, tau)] += q
        decisions.append(AssignmentDecision(customer_id=i, trip_id=b, station_id=h, planned_cost=cost))
    components = _cost_components(data, decisions)
    result = AssignmentResult(decisions=decisions, total_cost=components["total_objective"], method="greedy_fallback", solver_name="greedy_fallback", status="fallback_feasible", objective_value=components["total_objective"], used_fallback=True, fallback_reason=fallback_reason, number_assigned_customers=len(decisions), number_customers=len(data.customers), feasibility_summary={"all_customers_assigned_exactly_once": True}, cost_components=components)
    validate_assignment(data, result)
    return result


def solve_assignment(data, allow_greedy_fallback: bool = False, solver: str = "scipy", time_limit_sec: float | None = None, mip_gap: float | None = None, solver_disp: bool = False, solver_log: bool = True, auto_fallback_solver: str | None = None):
    model = build_assignment_model(data)
    idx = {k: i for i, k in enumerate(model.variable_keys)}
    n = len(model.variable_keys)
    _log(solver_log, f"[offline][solver] solver={solver} phase=build_model started")
    _log(solver_log, f"[offline][solver] solver={solver} phase=build_model completed variables={n} binary_variables={n} customers={len(data.customers)} trips={len(data.trips)} stations={len(data.stations)}")
    _log(solver_log, f"[offline][solver] solver={solver} phase=build_constraints started")
    constraints = _build_constraints(data, idx, n)
    _log(solver_log, f"[offline][solver] solver={solver} phase=build_constraints completed constraints={len(constraints)}")
    start = time.perf_counter()
    _log(solver_log, f"[offline][solver] solver={solver} phase=milp_solve started variables={n} constraints={len(constraints)} time_limit={time_limit_sec} mip_gap={mip_gap}")
    try:
        if solver == "gurobi":
            try:
                import gurobipy as gp
                from gurobipy import GRB
            except Exception as exc:
                msg = "gurobi unavailable: gurobipy is not installed or license is unavailable"
                _log(True, f"[offline][solver] {msg}")
                if auto_fallback_solver == "scipy":
                    _log(True, "[offline][solver] fallback from gurobi to scipy")
                    return solve_assignment(data, allow_greedy_fallback=allow_greedy_fallback, solver="scipy", time_limit_sec=time_limit_sec, mip_gap=mip_gap, solver_disp=solver_disp, solver_log=solver_log)
                raise SolverUnavailableError(msg) from exc
            m = gp.Model("offline_assignment")
            m.Params.OutputFlag = 1 if solver_disp else 0
            m.Params.LogToConsole = 1 if solver_disp else 0
            if time_limit_sec is not None: m.Params.TimeLimit = float(time_limit_sec)
            if mip_gap is not None: m.Params.MIPGap = float(mip_gap)
            x={k:m.addVar(vtype=GRB.BINARY,name=f"x_{k[0]}_{k[1]}_{k[2]}") for k in model.variable_keys}
            m.setObjective(gp.quicksum(model.objective[idx[k]]*x[k] for k in model.variable_keys), GRB.MINIMIZE)
            # constraints mirror
            for i in data.customers: m.addConstr(gp.quicksum(x[(b,h,i)] for h in data.feasible_stations_by_customer[i] for b in data.trips)==1)
            for b in data.trips: m.addConstr(gp.quicksum(data.parcel_weight[i]*x[(b,h,i)] for h in data.stations for i in data.feasible_customers_by_station[h])<=data.q_f[b])
            for b in data.trips:
                for h in data.stations: m.addConstr(gp.quicksum(data.parcel_weight[i]*x[(b,h,i)] for i in data.feasible_customers_by_station[h])<=data.q_u[h])
            for k in model.variable_keys:
                if data.delta_bh[(k[0],k[1])] == 0: m.addConstr(x[k] <= 0)
            for h in data.stations:
                for tau in data.t_grid: m.addConstr(gp.quicksum(data.parcel_weight[i]*x[(b,h,i)] for b in data.trips for i in data.feasible_customers_by_station[h] if data.r_bhi_0[(b,h,i)]<=tau<data.p_bhi_0[(b,h,i)])<=data.k[h])
            for h in data.stations: m.addConstr(gp.quicksum(data.t_hi_rt[(h,i)]*x[(b,h,i)] for b in data.trips for i in data.feasible_customers_by_station[h])<=data.num_drones[h]*data.operating_horizon)
            m.optimize()
            success = m.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}
            if not success or m.SolCount <= 0: raise AssignmentInfeasibleError(f"MILP failed: status={m.Status}")
            xv=[x[k].X for k in model.variable_keys]; obj=float(m.ObjVal); gap=float(getattr(m,'MIPGap',0.0)); status=str(m.Status)
        else:
            import numpy as np
            from scipy.optimize import Bounds, milp
            c = np.array(model.objective, dtype=float)
            integrality = np.ones(n, dtype=int)
            bounds = Bounds(np.zeros(n), np.ones(n))
            options = {}
            if time_limit_sec is not None: options['time_limit'] = float(time_limit_sec)
            if mip_gap is not None: options['mip_rel_gap'] = float(mip_gap)
            options['disp'] = bool(solver_disp)
            try:
                res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options=options)
            except TypeError:
                warnings.warn("SciPy milp options unsupported in this version; retrying without options.")
                res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            if not res.success: raise AssignmentInfeasibleError(f"MILP failed: {getattr(res,'message','unknown')}")
            xv=res.x; obj=float(res.fun); gap=getattr(res,'mip_gap',None); status=str(getattr(res,'message','unknown'))
        decisions=_extract_result(data, idx, xv)
        runtime=float(time.perf_counter()-start)
        _log(solver_log, f"[offline][solver] solver={solver} phase=milp_solve completed success=True status={status} objective={obj} gap={gap} runtime_sec={runtime}")
        _log(solver_log, f"[offline][solver] solver={solver} phase=extract_solution completed selected_variables={len(decisions)} assigned_customers={len(decisions)}/{len(data.customers)}")
        components=_cost_components(data, decisions)
        result=AssignmentResult(decisions=decisions,total_cost=components['total_objective'],method=f"{solver}_milp",solver_name=solver,status=status,objective_value=obj,runtime_sec=runtime,mip_gap=gap,used_fallback=False,number_assigned_customers=len(decisions),number_customers=len(data.customers),feasibility_summary={"all_customers_assigned_exactly_once": len(decisions)==len(data.customers)},cost_components=components,metadata={"solver": solver, "solver_status": status, "objective_value": obj, "mip_gap": gap, "runtime_sec": runtime, "time_limit_sec": time_limit_sec, "mip_gap_target": mip_gap, "assigned_customer_count": len(decisions), "total_customer_count": len(data.customers), "fallback_used": False, "fallback_reason": None})
        validate_assignment(data, result); return result
    except Exception as exc:
        runtime=float(time.perf_counter()-start)
        _log(True, f"[offline][solver] solver={solver} phase=milp_solve failed status=failed message={exc} runtime_sec={runtime}")
        if not allow_greedy_fallback:
            raise AssignmentInfeasibleError(f"MILP solve failed and greedy fallback is disabled: {exc}") from exc
        r=_greedy_fallback(data, fallback_reason=str(exc)); r.metadata={**(r.metadata or {}),"solver": solver, "fallback_used": True, "fallback_reason": str(exc), "runtime_sec": runtime, "time_limit_sec": time_limit_sec, "mip_gap_target": mip_gap, "assigned_customer_count": len(r.decisions), "total_customer_count": len(data.customers)}
        return r
