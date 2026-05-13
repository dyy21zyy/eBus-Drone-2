from __future__ import annotations

import numpy as np

from .action_space import action_index_to_duration, feasible_action_mask, feasible_actions, repair_action
from .bus_process import apply_charge, apply_travel_consumption
from .dwell_time import compute_dwell_breakdown
from .event_calendar import EventCalendar
from .parcel_process import compute_unloading_volume, get_unloading_parcels, unload_parcels_to_locker
from src.offline.assignment_io import build_assignment_indices
from .passenger_process import sample_poisson_arrivals, simulate_passenger_service_at_stop
from .reward import compute_reward
from src.low_level.station_operator import operate_station_step
from .termination import apply_terminal_penalty_once, check_termination
from .state_builder import build_observation


class EBusDroneEnv:
    def __init__(self, config: dict | None = None, instance: dict | None = None, scenario: dict | None = None, assignment: dict | None = None, smoke_test: bool = False):
        self.config = config or {}
        self.instance = instance
        self.scenario = scenario
        self.assignment = assignment
        self.smoke_test = smoke_test
        if not (self.instance and self.scenario and self.assignment) and not smoke_test:
            raise ValueError("EBusDroneEnv requires generated instance/scenario and offline assignment. Run generate then offline modes first.")
        self.terminal_penalty_applied = False
        self.reset()

    def reset(self, seed=None, options=None):
        _ = options
        if seed is not None:
            self.config["seed"] = int(seed)
        self._reset_from_data() if (self.instance and self.scenario and self.assignment) else self._reset_smoke()
        self.current_decision_event = None
        self.current_event = None
        self._advance_until_decision()
        self.terminal_penalty_applied = False
        return self._build_obs_for_current_event(), {"event": self.current_decision_event}

    def _reset_smoke(self):
        self.instance = {"horizon_minutes": 300.0, "network": {"stops": [{"stop_id": 0}, {"stop_id": 1}], "nominal_travel_time_min": [[0, 5], [5, 0]], "scheduled_bus_trips": [{"trip_id": 0, "departure_min": 0}]}, "stations": {"stations": [{"station_id": 1, "stop_id": 1, "chargers": 1, "drones": 1, "locker_capacity_kg": 100.0, "initial_fully_charged_batteries": 1, "initial_depleted_batteries": 0, "station_power_capacity_kw": 200.0}]}, "bus": {"battery_capacity_kwh": 150.0, "energy_kwh_per_km": 1.6}, "charging": {"pantograph_power_kw": 500.0, "charger_efficiency": 0.95}}
        self.scenario = {"passenger": {"passenger_arrivals": {"1": [0]}}}
        self.assignment = {"decisions": []}
        self._reset_from_data()

    def _reset_from_data(self):
        self.horizon = float(self.instance.get("horizon_minutes", 300.0))
        self.stop_ids = [int(s["stop_id"]) for s in self.instance["network"]["stops"]]
        self.station_by_stop = {int(s.get("stop_id", s["station_id"])): int(s["station_id"]) for s in self.instance["stations"]["stations"]}
        self.travel_times = self.instance["network"]["nominal_travel_time_min"][0]
        self.assignment_index = build_assignment_indices(self.assignment)
        self.calendar = EventCalendar()
        self.event_log = []
        self.rng = np.random.default_rng(int(self.config.get("seed", 0)))
        self.stop_queues = {sid: 0 for sid in self.stop_ids}
        self.stop_last_update = {sid: 0.0 for sid in self.stop_ids}

        cap = float(self.instance["bus"]["battery_capacity_kwh"])
        self.bus_states = {}
        for t in self.instance["network"]["scheduled_bus_trips"]:
            tid = int(t["trip_id"])
            pids = [int(d["customer_id"]) for d in self.assignment.get("decisions", []) if int(d["trip_id"]) == tid]
            self.bus_states[tid] = {"trip_id": tid, "current_stop_index": -1, "next_stop_id": self.stop_ids[0], "next_arrival_time_min": float(t["departure_min"]), "battery_kwh": cap, "battery_capacity_kwh": cap, "safety_battery_kwh": 5.0, "onboard_passengers": 0, "passenger_capacity": int(self.config.get("bus", {}).get("passenger_capacity", self.instance.get("bus", {}).get("passenger_capacity", 80))), "onboard_parcel_ids": pids, "active": True, "completed": False, "accumulated_delay_min": 0.0}
            self.calendar.add_bus_arrival(time=float(t["departure_min"]), trip_id=tid, stop_index=0, station_id=self.station_by_stop.get(self.stop_ids[0], -1), integrated=self.stop_ids[0] in self.station_by_stop, passengers_required=False, parcel_required=False)

        self.station_states = {}
        for s in self.instance["stations"]["stations"]:
            sid = int(s["station_id"])
            n_ch = int(s["chargers"])
            station_load_profile = self.scenario.get("power", {}).get("station_base_load_kw", {}).get(str(sid), self.scenario.get("power", {}).get("station_base_load_kw", {}).get(sid, []))
            self.station_states[sid] = {"station_id": sid, "charger_release_times_min": [0.0] * n_ch, "locker_parcels": [], "locker_parcel_ids": [], "locker_inventory_kg": 0.0, "locker_capacity_kg": float(s["locker_capacity_kg"]), "idle_drone_ids": [f"s{sid}_d{j}" for j in range(int(s["drones"]))], "active_drone_missions": [], "full_batteries": int(s["initial_fully_charged_batteries"]), "depleted_batteries": int(s["initial_depleted_batteries"]), "batteries_charging": [], "charging_batteries": [], "current_bus_charging_load": 0.0, "current_drone_battery_charging_load": 0.0, "power_capacity_kw": float(s["station_power_capacity_kw"]), "charging_slots": n_ch, "G_max": int(self.instance.get("battery", {}).get("max_simultaneous_charging", n_ch)), "P_capacity": float(s["station_power_capacity_kw"]), "P_bat": float(self.instance.get("battery", {}).get("charge_power_kw", 2.0)), "P_chg": float(self.instance["charging"]["pantograph_power_kw"]), "drones": [{"drone_id": f"s{sid}_d{j}", "status": "idle"} for j in range(int(s["drones"]))], "battery_charge_duration_min": float(self.instance.get("battery", {}).get("charge_duration_min", 10.0)), "base_load_profile_kw": station_load_profile, "base_load_fallback_kw": 50.0}

        self.parcel_states = {}
        customers = {int(c["customer_id"]): c for c in self.instance.get("customers", [])}
        for cid, t_id in self.assignment_index.get("by_customer", {}).items():
            c = customers.get(int(cid), {})
            assigned_station = int(self.assignment_index["station_by_customer"][cid])
            option = next((o for o in c.get("feasible_stations", []) if int(o["station_id"]) == assigned_station), None)
            if option is None:
                raise ValueError(f"Missing feasible station option for customer {cid} at station {assigned_station}")
            rt_min = float(option["mission_duration_min"])
            out_min = 0.5 * (rt_min - float(self.instance["drone"]["customer_service_time_min"]) - float(self.instance["drone"]["turnaround_time_min"]))
            self.parcel_states[cid] = {
                "parcel_id": int(cid), "id": int(cid), "customer_id": int(cid),
                "weight_kg": float(c.get("parcel_weight_kg", 0.0)),
                "delivery_deadline_min": float(c.get("delivery_deadline_min", self.horizon)),
                "deadline_min": float(c.get("delivery_deadline_min", self.horizon)),
                "assigned_trip_id": int(t_id), "assigned_station_id": assigned_station,
                "status": "onboard", "release_time_min": None, "pickup_time_min": None,
                "delivery_completion_time_min": None, "drone_return_time_min": None,
                "locker_holding_time_min": None, "T_out_min": out_min, "T_rt_min": rt_min,
                "T_out": out_min, "T_rt": rt_min,
                "drone_cost": float(self.instance["parcel"]["cost"]["drone_per_km"]) * float(option["distance_km"]),
            }

        self.delivered_parcels = set()
        self.state = {"time": 0.0, "horizon": self.horizon, "trip_location": 0, "battery": cap, "battery_max": cap, "onboard_passengers": 0, "onboard_parcels": 0, "queue": 0, "locker": 0, "idle_drones": 0, "full_batteries": 0, "station_power": 0.0, "power_margin": 0.0, "available_chargers": 0, "parcel_urgency": 0.0, "calendar_len": len(self.calendar), "charge_power_kw": float(self.instance["charging"]["pantograph_power_kw"]), "charge_efficiency": float(self.instance["charging"].get("charger_efficiency", 0.95)), "travel_energy_kwh_per_km": float(self.instance["bus"]["energy_kwh_per_km"]), "travel_distance_km": float(self.instance["network"].get("interstop_distance_km", 1.0)), "trip_id": 0}

    def _available_chargers(self, station_state, now):
        return sum(1 for t in station_state["charger_release_times_min"] if t <= now)
    def _active_bus_chargers(self, station_state, now):
        return sum(1 for t in station_state["charger_release_times_min"] if t > now)
    def _base_load_kw(self, station_state, now):
        prof = station_state.get("base_load_profile_kw", [])
        if prof:
            i = max(0, min(int(round(now)), len(prof) - 1))
            return float(prof[i])
        return float(station_state.get("base_load_fallback_kw", 50.0))

    def _advance_until_decision(self):
        self.current_decision_event = None
        self.current_event = None
        while len(self.calendar) > 0:
            e = self.calendar.pop_next()
            self.state["time"] = float(e.time)
            if e.kind == "charger_release":
                continue
            if e.kind == "dispatch_tick":
                st = self.station_states[e.station_id]
                p_e = self._active_bus_chargers(st, self.state["time"]) * self.state["charge_power_kw"]
                p_l = self._base_load_kw(st, self.state["time"])
                st["current_bus_charging_load"] = p_e
                operate_station_step(st, self.state["time"], parcel_states=self.parcel_states, delivered_parcels=self.delivered_parcels, p_e=p_e, p_l=p_l, new_parcels=False)
                continue
            if e.kind == "bus_arrival":
                if self._process_bus_arrival(e):
                    self.current_decision_event = e
                    self.current_event = e
                    return

    def _process_bus_arrival(self, e):
        bus = self.bus_states[int(e.trip_id)]
        if bus["completed"]:
            return False
        idx = int(e.stop_index)
        stop_id = self.stop_ids[idx]
        bus["current_stop_index"] = idx
        bus["next_stop_id"] = self.stop_ids[idx + 1] if idx + 1 < len(self.stop_ids) else -1

        station_id = self.station_by_stop.get(stop_id, -1)
        integrated = station_id != -1
        pax_required = False
        parcel_ids = get_unloading_parcels(bus["trip_id"], station_id, self.assignment_index, self.parcel_states) if integrated else []
        parcel_required = len(parcel_ids) > 0
        rate = float(self.scenario.get("passenger", {}).get("arrival_rate_per_stop_per_min", {}).get(str(stop_id), self.scenario.get("passenger", {}).get("arrival_rate_per_stop_per_min", {}).get(stop_id, 0.0)))
        al_p = float(self.scenario.get("passenger", {}).get("alighting_probability", 0.0))
        elapsed = max(0.0, float(e.time) - float(self.stop_last_update.get(stop_id, 0.0)))
        self.stop_queues[stop_id] += sample_poisson_arrivals(rate, elapsed, self.rng)
        self.stop_last_update[stop_id] = float(e.time)
        service = simulate_passenger_service_at_stop(queue=self.stop_queues[stop_id], onboard=bus["onboard_passengers"], capacity=bus["passenger_capacity"], alighting_probability=al_p, rate_per_min=rate, rho_al_min_per_pax=1.5/60.0, rho_bo_min_per_pax=3.0/60.0, parcel_unloading_time_min=0.0, charging_duration_min=0.0, rng=self.rng)
        bus["onboard_passengers"] = service["onboard_final"]
        self.stop_queues[stop_id] = service["queue_final"]
        pax_required = bool(service["chi"])
        requires_stop = pax_required or parcel_required

        if requires_stop and integrated:
            e.station_id, e.integrated, e.passengers_required, e.parcel_required, e.requires_stop, e.is_decision = station_id, True, pax_required, parcel_required, True, True
            e.passenger_service = service
            return True

        dwell = service["realized_dwell_min"] if requires_stop else 0.0
        dep = e.time + dwell
        if idx + 1 < len(self.stop_ids):
            next_t = dep + float(self.travel_times[idx + 1] - self.travel_times[idx])
            self.calendar.add_bus_arrival(time=next_t, trip_id=bus["trip_id"], stop_index=idx + 1, station_id=self.station_by_stop.get(self.stop_ids[idx+1], -1), integrated=self.stop_ids[idx+1] in self.station_by_stop, passengers_required=False, parcel_required=False)
            bus["next_arrival_time_min"] = next_t
            bus["battery_kwh"] = apply_travel_consumption(bus["battery_kwh"], self.state["travel_distance_km"], self.state["travel_energy_kwh_per_km"])
        else:
            bus["completed"] = True
            bus["active"] = False
        return False

    def get_action_mask(self) -> np.ndarray:
        if self.current_decision_event is None:
            return np.zeros(6, dtype=np.int32)
        bus = self.bus_states[int(self.current_decision_event.trip_id)]
        st = self.station_states[int(self.current_decision_event.station_id)]
        return feasible_action_mask(self._available_chargers(st, self.state["time"]), bus["battery_kwh"], bus["battery_capacity_kwh"], self.state["charge_power_kw"], self.state["charge_efficiency"])
    def get_feasible_actions(self): return feasible_actions(self.get_action_mask())
    def repair_action(self, action_index): return repair_action(action_index, self.get_action_mask())

    def step(self, action_index):
        e = self.current_decision_event
        if e is None:
            self.current_event = None
            term, reason = check_termination(self.state, False)
            return self._build_obs_for_current_event(), 0.0, term, False, {"termination_reason": reason}
        bus = self.bus_states[int(e.trip_id)]
        st = self.station_states[int(e.station_id)]
        mask = self.get_action_mask(); ex_idx = action_index if mask[action_index] else self.repair_action(action_index)
        dur = action_index_to_duration(ex_idx)
        before = bus["battery_kwh"]
        av_before = self._available_chargers(st, self.state["time"])
        if dur > 0 and av_before > 0:
            i = next(i for i, t in enumerate(st["charger_release_times_min"]) if t <= self.state["time"])
            st["charger_release_times_min"][i] = self.state["time"] + dur / 60.0
            self.calendar.add_charger_release(time=st["charger_release_times_min"][i], station_id=st["station_id"])
            st["current_bus_charging_load"] = self._active_bus_chargers(st, self.state["time"]) * self.state["charge_power_kw"]
        bus["battery_kwh"] = apply_charge(bus["battery_kwh"], dur, self.state["charge_power_kw"], self.state["charge_efficiency"], bus["battery_capacity_kwh"])

        unload_ids = get_unloading_parcels(bus["trip_id"], st["station_id"], self.assignment_index, self.parcel_states)
        qf = compute_unloading_volume(unload_ids, self.parcel_states)
        unloaded = unload_parcels_to_locker(bus, st, self.parcel_states, unload_ids, self.state["time"]) if unload_ids else []
        stop_id = self.stop_ids[int(e.stop_index)]
        rate = float(self.scenario.get("passenger", {}).get("arrival_rate_per_stop_per_min", {}).get(str(stop_id), self.scenario.get("passenger", {}).get("arrival_rate_per_stop_per_min", {}).get(stop_id, 0.0)))
        al_p = float(self.scenario.get("passenger", {}).get("alighting_probability", 0.0))
        elapsed = max(0.0, float(self.state["time"]) - float(self.stop_last_update.get(stop_id, 0.0)))
        self.stop_queues[stop_id] += sample_poisson_arrivals(rate, elapsed, self.rng)
        self.stop_last_update[stop_id] = float(self.state["time"])
        service = simulate_passenger_service_at_stop(queue=self.stop_queues[stop_id], onboard=bus["onboard_passengers"], capacity=bus["passenger_capacity"], alighting_probability=al_p, rate_per_min=rate, rho_al_min_per_pax=1.5/60.0, rho_bo_min_per_pax=3.0/60.0, parcel_unloading_time_min=qf*0.5, charging_duration_min=dur/60.0, rng=self.rng)
        bus["onboard_passengers"] = service["onboard_final"]
        self.stop_queues[stop_id] = service["queue_final"]

        dep = self.state["time"] + service["realized_dwell_min"]
        idx = int(e.stop_index)
        if idx + 1 < len(self.stop_ids):
            next_t = dep + float(self.travel_times[idx + 1] - self.travel_times[idx])
            self.calendar.add_bus_arrival(time=next_t, trip_id=bus["trip_id"], stop_index=idx + 1, station_id=self.station_by_stop.get(self.stop_ids[idx+1], -1), integrated=self.stop_ids[idx+1] in self.station_by_stop, passengers_required=False, parcel_required=False)
            bus["next_arrival_time_min"] = next_t
            bus["battery_kwh"] = apply_travel_consumption(bus["battery_kwh"], self.state["travel_distance_km"], self.state["travel_energy_kwh_per_km"])
        else:
            bus["completed"] = True; bus["active"] = False

        p_e = self._active_bus_chargers(st, dep) * self.state["charge_power_kw"]
        p_l = self._base_load_kw(st, dep)
        st["current_bus_charging_load"] = p_e
        op = operate_station_step(st, dep, parcel_states=self.parcel_states, delivered_parcels=self.delivered_parcels, p_e=p_e, p_l=p_l, new_parcels=bool(unloaded))
        self.event_log.append({"time_min": self.state["time"], "bus_id": bus["trip_id"], "station_id": st["station_id"], "action_sec": dur, "dwell_time_min": service["realized_dwell_min"], "passenger_delay": float(service["total_board"]), "parcel_unloading_kg": qf, "battery_before": before, "battery_after_charge": apply_charge(before, dur, self.state["charge_power_kw"], self.state["charge_efficiency"], bus["battery_capacity_kwh"]), "available_chargers_before": av_before, "station_power_overload": float(op["power"].get("overload", 0.0))})

        self.state.update({"time": dep, "trip_id": bus["trip_id"], "trip_location": idx, "battery": bus["battery_kwh"], "battery_max": bus["battery_capacity_kwh"], "onboard_passengers": bus["onboard_passengers"], "onboard_parcels": len(bus["onboard_parcel_ids"]), "locker": int(st["locker_inventory_kg"]), "idle_drones": sum(1 for d in st.get("drones", []) if d.get("status") == "idle"), "full_batteries": st.get("full_batteries", 0), "station_power": op["P_tot"], "power_margin": st["power_capacity_kw"] - op["P_tot"], "available_chargers": self._available_chargers(st, dep), "calendar_len": len(self.calendar)})
        comp = {"D_P": float(service["total_board"]), "D_L": 0.0, "D_E": 0.0, "D_Pwr": float(op["power"].get("overload", 0.0)), "D_B": max(0.0, bus["safety_battery_kwh"] - bus["battery_kwh"]), "D_K": max(0.0, st["locker_inventory_kg"] - st["locker_capacity_kg"])}
        reward, rc = compute_reward(comp, {"alpha_1": 0.01, "alpha_2": 1.0, "alpha_3": 1.0, "alpha_4": 1.0, "alpha_5": 1.0, "alpha_6": 1.0})
        self._advance_until_decision()
        terminated, reason = check_termination(self.state, self.current_decision_event is not None)
        if terminated:
            terminal_penalty = apply_terminal_penalty_once(self.__dict__, [p for p in self.parcel_states.values() if p.get("status") != "delivered"], self.state["time"], 1.0, 1.0)
            rc["terminal_penalty"] = terminal_penalty
        return self._build_obs_for_current_event(), float(reward), terminated, False, {"executed_action_index": ex_idx, "executed_duration": dur, "action_repaired": ex_idx != action_index, "termination_reason": reason, "reward_components": rc, "unloaded_parcels": unloaded, "unloading_volume_kg": qf, "current_trip_id": bus["trip_id"], "current_station_id": st["station_id"]}

    def _build_obs_for_current_event(self):
        e = self.current_decision_event
        if e is None:
            local = {"station_id": -1, "trip_id": self.state.get("trip_id", -1), "arriving_battery": self.state.get("battery", 0.0), "onboard_before_alight": self.state.get("onboard_passengers", 0), "onboard_parcels_before_unload": self.state.get("onboard_parcels", 0), "alighting": 0, "initial_board": 0, "q_f": 0.0, "available_chargers": 0, "locker": self.state.get("locker", 0), "idle_drones": self.state.get("idle_drones", 0), "full_batteries": self.state.get("full_batteries", 0), "power_margin": self.state.get("power_margin", 0.0), "delivery_urgency": self.state.get("parcel_urgency", 0.0)}
            return build_observation(self.state, local)
        bus = self.bus_states[int(e.trip_id)]; st = self.station_states[int(e.station_id)]
        unload_ids = get_unloading_parcels(bus["trip_id"], st["station_id"], self.assignment_index, self.parcel_states)
        local = {"station_id": st["station_id"], "trip_id": bus["trip_id"], "arriving_battery": bus["battery_kwh"], "onboard_before_alight": bus["onboard_passengers"], "onboard_parcels_before_unload": len(bus["onboard_parcel_ids"]), "onboard_parcel_weight_before_unload": sum(self.parcel_states[p]["weight_kg"] for p in bus["onboard_parcel_ids"] if p in self.parcel_states), "alighting": 0, "initial_board": 0, "q_f": compute_unloading_volume(unload_ids, self.parcel_states), "num_unloading_parcels": len(unload_ids), "available_chargers": self._available_chargers(st, self.state["time"]), "locker": st["locker_inventory_kg"], "idle_drones": sum(1 for d in st.get("drones", []) if d.get("status") == "idle"), "full_batteries": st.get("full_batteries", 0), "power_margin": st["power_capacity_kw"] - st.get("current_bus_charging_load", 0.0), "delivery_urgency": self.state.get("parcel_urgency", 0.0)}
        return build_observation(self.state, local)
