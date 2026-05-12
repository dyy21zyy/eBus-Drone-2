from __future__ import annotations

from .battery_manager import add_depleted_batteries, charge_depleted_batteries, consume_full_batteries
from .dispatch_trigger import should_trigger_dispatch
from .drone_dispatch_solver import solve_greedy_dispatch
from .drone_manager import idle_drone_ids, mark_active, process_returns
from .locker_manager import feasible_parcels, remove_parcels_by_id
from .station_power import compute_station_power


def operate_station_step(station_state: dict, now: float, *, p_e: float, p_l: float, eta_l_d: float = 1.0, eta_u_d: float = 1.0, dispatch_interval: float = 5.0, new_parcels: bool = False) -> dict:
    returned = process_returns(station_state, now)
    if returned:
        add_depleted_batteries(station_state, returned)

    charge = charge_depleted_batteries(station_state, p_e=p_e, p_l=p_l)
    battery_charged = charge["g"] > 0

    trigger = should_trigger_dispatch(
        new_parcels=new_parcels,
        drone_returned=returned > 0,
        battery_fully_charged=battery_charged,
        t=now,
        dispatch_interval=dispatch_interval,
    )
    assignments = []
    n_disp = 0
    if trigger:
        waiting = station_state.get("locker_parcels", [])
        feas = feasible_parcels(waiting, now)
        idle = idle_drone_ids(station_state)
        assignments, n_disp = solve_greedy_dispatch(idle, station_state.get("full_batteries", 0), feas, now, eta_l_d, eta_u_d)
        if n_disp > 0:
            consume_full_batteries(station_state, n_disp)
            assigned_ids = [str(a["parcel"]["id"]) for a in assignments]
            station_state["locker_parcels"] = remove_parcels_by_id(waiting, assigned_ids)
            for a in assignments:
                p = a["parcel"]
                p["pickup_time"] = now
                p["completion_time"] = now + float(p["T_out"])
                p["return_time"] = now + float(p["T_rt"])
            for a in assignments:
                mark_active(station_state, [a["drone_id"]], a["parcel"]["return_time"])

    power = compute_station_power(p_e, p_l, charge["P_D"], station_state.get("P_capacity", 0.0))
    return {"triggered": trigger, "n_disp": n_disp, "assignments": assignments, "charge": charge, "power": power, "returns": returned}
