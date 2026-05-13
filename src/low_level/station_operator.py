from __future__ import annotations

from .battery_manager import add_depleted_batteries, charge_depleted_batteries, consume_full_batteries
from .dispatch_trigger import should_trigger_dispatch
from .drone_dispatch_solver import solve_greedy_dispatch
from .drone_manager import idle_drone_ids, mark_active, process_returns
from .locker_manager import feasible_parcels
from .station_power import compute_station_power


def operate_station_step(station_state: dict, now: float, *, parcel_states: dict | None = None, delivered_parcels: set[int] | None = None, p_e: float, p_l: float, eta_l_d: float = 1.0, eta_u_d: float = 1.0, dispatch_interval: float = 5.0, new_parcels: bool = False, max_round_trip_duration: float = 1e9) -> dict:
    parcel_states = parcel_states or {}
    delivered_ids: list[int] = []
    for p in parcel_states.values():
        completion_time = p.get("delivery_completion_time_min", p.get("delivery_completion_time"))
        if p.get("status") == "assigned_to_drone" and completion_time is not None and float(completion_time) <= now:
            p["status"] = "delivered"
            p["delivery_completion_time_min"] = float(completion_time)
            p["lateness"] = max(0.0, float(completion_time) - float(p.get("delivery_deadline_min", p.get("deadline_min", now))))
            delivered_ids.append(int(p["parcel_id"]))
            if delivered_parcels is not None:
                delivered_parcels.add(int(p["parcel_id"]))

    returned = process_returns(station_state, now)
    if returned:
        add_depleted_batteries(station_state, len(returned))

    charge = charge_depleted_batteries(station_state, now=now, p_e=p_e, p_l=p_l)
    battery_charged = charge["completed"] > 0

    trigger = should_trigger_dispatch(
        new_parcels=new_parcels,
        drone_returned=len(returned) > 0,
        battery_fully_charged=battery_charged,
        t=now,
        dispatch_interval=dispatch_interval,
    )
    assignments = []
    n_disp = 0
    if trigger:
        waiting = [parcel_states[int(pid)] for pid in station_state.get("locker_parcels", []) if int(pid) in parcel_states]
        feas = feasible_parcels(waiting, now, int(station_state.get("station_id", -1)), max_round_trip_duration)
        idle = idle_drone_ids(station_state)
        assignments, n_disp = solve_greedy_dispatch(idle, station_state.get("full_batteries", 0), feas, now, eta_l_d, eta_u_d)
        if n_disp > 0:
            consume_full_batteries(station_state, n_disp)
            assigned_ids = [int(a["parcel_id"]) for a in assignments]
            station_state["locker_parcels"] = [int(pid) for pid in station_state.get("locker_parcels", []) if int(pid) not in set(assigned_ids)]
            for a in assignments:
                p = parcel_states[int(a["parcel_id"])]
                p["status"] = "assigned_to_drone"
                p["pickup_time_min"] = now
                p["pickup_time"] = now
                p["drone_id"] = a["drone_id"]
                t_out = float(p.get("T_out_min", p.get("T_out", 0.0)))
                t_rt = float(p.get("T_rt_min", p.get("T_rt", 0.0)))
                p["delivery_completion_time_min"] = now + t_out
                p["delivery_completion_time"] = p["delivery_completion_time_min"]
                p["drone_return_time_min"] = now + t_rt
                p["drone_return_time"] = p["drone_return_time_min"]
                if p.get("release_time_min") is not None:
                    p["locker_holding_time_min"] = now - float(p["release_time_min"])
                a["drone_return_time"] = p["drone_return_time_min"]
            mark_active(station_state, assignments)

    station_state["locker_inventory_kg"] = float(sum(float(parcel_states[int(pid)]["weight_kg"]) for pid in station_state.get("locker_parcels", []) if int(pid) in parcel_states and parcel_states[int(pid)].get("status") == "in_locker"))

    power = compute_station_power(p_e, p_l, charge["P_D"], station_state.get("P_capacity", 0.0))
    station_state["current_drone_battery_charging_load"] = power["P_D"]
    dt = max(0.0, float(now) - float(station_state.get("last_power_update_time_min", now)))
    station_state["last_power_update_time_min"] = float(now)
    overload_kw = float(power["overload"])
    station_state["bus_charging_energy_kwh"] = float(station_state.get("bus_charging_energy_kwh", 0.0)) + float(p_e) * dt / 60.0
    station_state["drone_charging_energy_kwh"] = float(station_state.get("drone_charging_energy_kwh", 0.0)) + float(power["P_D"]) * dt / 60.0
    station_state["total_energy_kwh"] = station_state["bus_charging_energy_kwh"] + station_state["drone_charging_energy_kwh"]
    station_state["power_overload_amount_kw_min"] = float(station_state.get("power_overload_amount_kw_min", 0.0)) + overload_kw * dt
    station_state["power_overload_duration_min"] = float(station_state.get("power_overload_duration_min", 0.0)) + (dt if overload_kw > 0 else 0.0)
    total_chargers = max(1, len(station_state.get("charger_release_times_min", [])))
    active_chargers = int(round(float(p_e) / max(1e-9, float(station_state.get("P_chg", p_e if p_e > 0 else 1.0)))))
    station_state["charger_utilization"] = float(active_chargers) / float(total_chargers)
    return {"triggered": trigger, "n_disp": n_disp, "assignments": assignments, "dispatched_parcel_ids": [a["parcel_id"] for a in assignments], "delivered_parcel_ids": delivered_ids, "drone_return_ids": returned, "batteries_charged": charge["completed"], "full_batteries": station_state.get("full_batteries", 0), "depleted_batteries": station_state.get("depleted_batteries", station_state.get("empty_batteries", 0)), "P_D": power["P_D"], "P_tot": power["P_tot"], "overload": power["overload"], "charge": charge, "power": power, "metrics": {"bus_charging_energy_kwh": station_state["bus_charging_energy_kwh"], "drone_charging_energy_kwh": station_state["drone_charging_energy_kwh"], "total_energy_kwh": station_state["total_energy_kwh"], "power_overload_amount_kw_min": station_state["power_overload_amount_kw_min"], "power_overload_duration_min": station_state["power_overload_duration_min"], "charger_utilization": station_state["charger_utilization"]}}
