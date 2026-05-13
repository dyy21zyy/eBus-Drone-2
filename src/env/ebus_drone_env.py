from __future__ import annotations

import numpy as np

from .action_space import action_index_to_duration, feasible_action_mask, feasible_actions, repair_action
from .bus_process import apply_charge, apply_travel_consumption
from .charger_process import occupy_charger, release_charger
from .dwell_time import compute_dwell_breakdown
from .event_calendar import EventCalendar
from .parcel_process import compute_unloading_volume, get_unloading_parcels, unload_parcels_to_locker
from src.offline.assignment_io import build_assignment_indices
from .passenger_process import simulate_passenger_stop
from .reward import compute_reward
from src.low_level.station_operator import operate_station_step
from .termination import apply_terminal_penalty_once, check_termination
from .state_builder import build_observation
from .transition import advance_time


class EBusDroneEnv:
    def __init__(self, config: dict | None = None, instance: dict | None = None, scenario: dict | None = None, assignment: dict | None = None, smoke_test: bool = False):
        self.config = config or {}
        self.instance = instance
        self.scenario = scenario
        self.assignment = assignment
        self.smoke_test = smoke_test
        if not (self.instance and self.scenario and self.assignment):
            if not smoke_test:
                raise ValueError("EBusDroneEnv requires generated instance/scenario and offline assignment. Run generate then offline modes first.")
        self.terminal_penalty_applied = False
        self.reset()

    def reset(self, seed=None, options=None):
        _ = seed, options
        if self.instance and self.scenario and self.assignment:
            self._reset_from_data()
        else:
            self._reset_smoke()
        self.current_event = self._next_decision_event()
        self.terminal_penalty_applied = False
        return self._build_obs_for_current_event(), {"event": self.current_event}

    def _reset_smoke(self):
        self.state = {"time": 0.0, "horizon": 300.0, "battery": 120.0, "battery_max": 150.0, "charge_power_kw": 500.0, "charge_efficiency": 0.95, "travel_energy_kwh_per_km": 1.6, "travel_distance_km": 10.0, "travel_time": 5.0, "trip_location": 0, "onboard_passengers": 12, "onboard_parcels": 6, "queue": 4, "locker": 0, "idle_drones": 2, "full_batteries": 3, "empty_batteries": 0, "station_power": 500.0, "power_margin": 100.0, "available_chargers": 1, "total_chargers": 1, "parcel_urgency": 0.3, "infeasible": False, "trip_id": 0}
        self.assignment_index = {"by_trip_station": {}, "by_customer": {}, "station_by_customer": {}}
        self.parcel_states = {}
        self.bus_states = {0: {"trip_id": 0, "onboard_parcel_ids": [], "onboard_parcel_weight": 0.0}}
        self.station_states = {}
        self.delivered_parcels = set()
        self.calendar = EventCalendar()

    def _reset_from_data(self):
        horizon = float(self.instance.get("horizon_minutes", self.config.get("generation", {}).get("horizon_minutes", 300)))
        battery_max = float(self.instance["bus"]["battery_capacity_kwh"])
        charge_power_kw = float(self.instance["charging"]["pantograph_power_kw"])
        station0 = self.instance["stations"]["stations"][0]
        decisions = self.assignment.get("decisions", [])
        self.assignment_index = build_assignment_indices(self.assignment)
        trip_id = int(self.instance["network"]["scheduled_bus_trips"][0]["trip_id"])
        onboard = [d for d in decisions if int(d["trip_id"]) == trip_id]
        self.parcel_states = {}
        weights = {int(c["customer_id"]): float(c["parcel_weight_kg"]) for c in self.instance["customers"]}
        for cid, t_id in self.assignment_index["by_customer"].items():
            self.parcel_states[cid] = {"parcel_id": cid, "id": cid, "customer_id": cid, "weight_kg": weights[cid], "status": "onboard", "current_trip_id": int(t_id), "assigned_trip_id": int(t_id), "assigned_station_id": int(self.assignment_index["station_by_customer"][cid]), "deadline_min": float(self.instance.get("horizon_minutes", 300.0)), "release_time": None, "pickup_time": None, "delivery_completion_time": None, "drone_return_time": None, "lateness": None, "drone_id": None, "station_id": None, "T_out": 5.0, "T_rt": 10.0, "c_D": 1.0}

        self.station_states = {int(s["station_id"]): {"station_id": int(s["station_id"]), "locker_parcels": [], "locker_inventory_kg": 0.0, "locker_capacity_kg": float(s["locker_capacity_kg"]), "drones": [{"drone_id": f"s{int(s["station_id"])}_d{j}", "status": "idle", "assigned_parcel_id": None, "return_time": None, "home_station_id": int(s["station_id"])} for j in range(int(s["drones"]))], "full_batteries": int(s["initial_fully_charged_batteries"]), "depleted_batteries": int(s["initial_depleted_batteries"]), "empty_batteries": int(s["initial_depleted_batteries"]), "charging_batteries": [], "charging_slots": int(s["chargers"]), "G_max": int(s["chargers"]), "P_capacity": float(s["station_power_capacity_kw"]), "station_power_capacity_kw": float(s["station_power_capacity_kw"]), "current_base_load_kw": 50.0, "current_bus_charging_load_kw": 20.0, "current_drone_charging_load_kw": 0.0, "P_bat": 10.0, "battery_charge_duration_min": 10.0, "max_round_trip_duration": 120.0} for s in self.instance["stations"]["stations"]}
        self.state = {"time": 0.0, "horizon": horizon, "battery": battery_max, "battery_max": battery_max, "charge_power_kw": charge_power_kw, "charge_efficiency": float(self.instance["charging"].get("charger_efficiency", 0.95)), "travel_energy_kwh_per_km": float(self.instance["bus"]["energy_kwh_per_km"]), "travel_distance_km": float(self.instance["network"].get("interstop_distance_km", 10.0)), "travel_time": 5.0, "trip_location": 0, "onboard_passengers": 0, "onboard_parcels": len(onboard), "queue": 0, "locker": 0, "idle_drones": int(station0["drones"]), "full_batteries": int(station0["initial_fully_charged_batteries"]), "empty_batteries": int(station0["initial_depleted_batteries"]), "station_power": float(station0["station_power_capacity_kw"]), "power_margin": 100.0, "available_chargers": int(station0["chargers"]), "total_chargers": int(station0["chargers"]), "parcel_urgency": 0.0, "infeasible": False, "trip_id": trip_id}
        self.bus_states = {int(t["trip_id"]): {"trip_id": int(t["trip_id"]), "current_stop_index": 0, "next_event_time": float(t["departure_min"]), "battery": battery_max, "onboard_passenger_load": 0, "onboard_parcel_ids": [int(d["customer_id"]) for d in decisions if int(d["trip_id"]) == int(t["trip_id"])], "onboard_parcel_weight": 0.0, "active": True} for t in self.instance["network"]["scheduled_bus_trips"]}
        for t_id, trip_state in self.bus_states.items():
            trip_state["onboard_parcel_weight"] = float(sum(self.parcel_states[pid]["weight_kg"] for pid in trip_state["onboard_parcel_ids"]))
        self.delivered_parcels = set()
        self.calendar = EventCalendar(); self.calendar.build_from_generated(self.instance, self.scenario, self.assignment)

    def get_action_mask(self) -> np.ndarray:
        return feasible_action_mask(self.state["available_chargers"], self.state["battery"], self.state["battery_max"], self.state["charge_power_kw"], self.state["charge_efficiency"])
    def get_feasible_actions(self) -> list[int]: return feasible_actions(self.get_action_mask())
    def repair_action(self, action_index) -> int: return repair_action(action_index, self.get_action_mask())

    def step(self, action_index):
        selected_duration = action_index_to_duration(action_index); mask = self.get_action_mask(); executed_action_index = action_index if mask[action_index] else self.repair_action(action_index); executed_duration = action_index_to_duration(executed_action_index)
        self.state["battery"] = apply_charge(self.state["battery"], executed_duration, self.state["charge_power_kw"], self.state["charge_efficiency"], self.state["battery_max"]); self.state["available_chargers"] = occupy_charger(self.state["available_chargers"], executed_duration)
        event = self.current_event
        trip_id = -1 if event is None else int(event.trip_id)
        station_id = -1 if event is None else int(event.station_id)
        trip_state = self.bus_states.get(trip_id)
        station_state = self.station_states.get(station_id)
        pax_required = bool(event.passengers_required) if event else False
        pax = simulate_passenger_stop(self.state["queue"], self.state["onboard_passengers"], 40, 0, arrivals_during=1 if pax_required else 0); self.state["onboard_passengers"] = pax["onboard_final"]; self.state["queue"] = pax["queue_final"]
        u_r = [] if (trip_state is None or station_state is None) else get_unloading_parcels(trip_id, station_id, self.assignment_index, self.parcel_states)
        q_f = compute_unloading_volume(u_r, self.parcel_states)
        unloaded_ids = [] if (trip_state is None or station_state is None) else unload_parcels_to_locker(trip_state, station_state, self.parcel_states, u_r, self.state["time"])
        self.state["onboard_parcels"] = len(trip_state["onboard_parcel_ids"]) if trip_state else 0
        self.state["locker"] = int(station_state["locker_inventory_kg"]) if station_state else self.state["locker"]
        dwell = compute_dwell_breakdown(0, pax["initial_board"], 1.0, 1.0, q_f, executed_duration, 0.5, 0.5, 1.0, self.state["onboard_passengers"], pax_required)
        self.state["time"] = advance_time(self.current_event.time if self.current_event else self.state["time"], dwell.t_s, self.state["travel_time"]); self.state["battery"] = apply_travel_consumption(self.state["battery"], self.state["travel_distance_km"], self.state["travel_energy_kwh_per_km"]); self.state["available_chargers"] = release_charger(self.state["available_chargers"], self.state["total_chargers"])
        self.current_event = self._next_decision_event(); terminated, reason = check_termination(self.state, self.current_event is not None)
        op = {"power": {"overload": 0.0}}
        for sid, st in self.station_states.items():
            station_op = operate_station_step(st, self.state["time"], parcel_states=self.parcel_states, delivered_parcels=self.delivered_parcels, p_e=st.get("current_base_load_kw", 50.0), p_l=st.get("current_bus_charging_load_kw", 20.0), new_parcels=bool(unloaded_ids and station_state is not None and sid == int(station_state["station_id"])), max_round_trip_duration=st.get("max_round_trip_duration", 120.0))
            st["current_drone_charging_load_kw"] = station_op["P_D"]
            if station_state is not None and sid == int(station_state["station_id"]):
                op = station_op
                self.state["full_batteries"] = int(st.get("full_batteries", self.state["full_batteries"]))
                self.state["empty_batteries"] = int(st.get("depleted_batteries", self.state["empty_batteries"]))
                self.state["idle_drones"] = sum(1 for d in st.get("drones", []) if d.get("status") == "idle")
        locker_over = 0.0
        if station_state is not None and station_state["locker_inventory_kg"] > station_state["locker_capacity_kg"]:
            locker_over = station_state["locker_inventory_kg"] - station_state["locker_capacity_kg"]
        transition_minutes = max(0.0, dwell.t_s + self.state["travel_time"])
        bus_energy_kwh = self.state["charge_efficiency"] * self.state["charge_power_kw"] * executed_duration / 3600.0
        drone_energy_kwh = max(0.0, float(op.get("P_D", 0.0))) * transition_minutes / 60.0
        d_l = 0.0
        for pid in list(self.delivered_parcels):
            if pid in self.delivered_parcels:
                pr = self.parcel_states.get(pid, {})
                c_i = float(pr.get("delivery_completion_time", self.state["time"]))
                d_i = float(pr.get("deadline_min", self.state["horizon"]))
                d_l += max(0.0, c_i - d_i)
        d_b = max(0.0, 5.0 - float(self.state.get("battery", 0.0)))
        components = {"D_P": dwell.passenger_delay, "D_L": d_l, "D_E": (bus_energy_kwh + drone_energy_kwh), "D_Pwr": float(op["power"].get("overload", 0.0))*transition_minutes, "D_B": d_b, "D_K": locker_over*transition_minutes}
        terminal_penalty = 0.0
        if terminated:
            undelivered = [p for p in self.parcel_states.values() if p.get("status") != "delivered"]
            terminal_penalty = apply_terminal_penalty_once(self.__dict__, undelivered, self.state["time"], 1.0, 1.0)
            components["D_L"] += terminal_penalty
        reward, rc = compute_reward(components, {"alpha_1": 0.01, "alpha_2": 1.0, "alpha_3": 1.0, "alpha_4": 1.0, "alpha_5": 1.0, "alpha_6": 1.0})
        rc.update({"terminal_penalty": terminal_penalty, "total_cost": -reward, "reward": reward})
        return self._build_obs_for_current_event(), float(reward), terminated, False, {"executed_action_index": executed_action_index, "selected_duration": selected_duration, "executed_duration": executed_duration, "action_repaired": executed_action_index != action_index, "termination_reason": reason, "reward_components": rc, "unloaded_parcels": unloaded_ids, "unloading_volume_kg": q_f, "current_trip_id": trip_id, "current_station_id": station_id}

    def _next_decision_event(self):
        while len(self.calendar) > 0:
            e = self.calendar.pop_next()
            if e and e.is_decision and e.integrated and e.requires_stop:
                return e
        return None

    def _build_obs_for_current_event(self):
        e = self.current_event
        trip_id = self.state["trip_id"] if e is None else int(e.trip_id)
        station_id = -1 if e is None else int(e.station_id)
        trip_state = self.bus_states.get(trip_id, {"onboard_parcel_weight": 0.0})
        unload_ids = [] if e is None else get_unloading_parcels(trip_id, station_id, self.assignment_index, self.parcel_states)
        q_f = compute_unloading_volume(unload_ids, self.parcel_states)
        locker_inv = self.station_states.get(station_id, {}).get("locker_inventory_kg", self.state["locker"])
        local = {"station_id": station_id, "trip_id": trip_id, "arriving_battery": self.state["battery"], "onboard_before_alight": self.state["onboard_passengers"], "onboard_parcels_before_unload": self.state["onboard_parcels"], "onboard_parcel_weight_before_unload": trip_state.get("onboard_parcel_weight", 0.0), "alighting": 0, "initial_board": min(self.state["queue"], 40 - self.state["onboard_passengers"]), "q_f": q_f, "num_unloading_parcels": len(unload_ids), "available_chargers": self.state["available_chargers"], "locker": locker_inv, "idle_drones": self.state["idle_drones"], "full_batteries": self.state["full_batteries"], "power_margin": self.state["power_margin"], "delivery_urgency": self.state["parcel_urgency"]}
        self.state["calendar_len"] = len(self.calendar)
        return build_observation(self.state, local)
