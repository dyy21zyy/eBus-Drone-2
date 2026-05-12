from __future__ import annotations

import numpy as np

from .action_space import A_FULL, action_index_to_duration, feasible_action_mask, feasible_actions, repair_action
from .bus_process import apply_charge, apply_travel_consumption
from .charger_process import occupy_charger, release_charger
from .dwell_time import compute_dwell_breakdown
from .event_calendar import EventCalendar
from .event_types import Event
from .parcel_process import unload_parcels
from .passenger_process import simulate_passenger_stop
from .reward import compute_reward
from .state_builder import build_observation
from .termination import check_termination
from .transition import advance_time


class EBusDroneEnv:
    def __init__(self):
        self.terminal_penalty_applied = False
        self.reset()

    def reset(self, seed=None, options=None):
        _ = seed, options
        self.state = {
            "time": 0.0,
            "horizon": 300.0,
            "battery": 120.0,
            "battery_max": 150.0,
            "charge_rate": 0.5,
            "travel_consumption": 6.0,
            "travel_time": 5.0,
            "trip_location": 0,
            "onboard_passengers": 12,
            "onboard_parcels": 6,
            "queue": 4,
            "locker": 0,
            "idle_drones": 2,
            "full_batteries": 3,
            "station_power": 500.0,
            "power_margin": 100.0,
            "available_chargers": 1,
            "total_chargers": 1,
            "parcel_urgency": 0.3,
            "infeasible": False,
            "trip_id": 0,
        }
        self.calendar = EventCalendar()
        self.calendar.add(Event(0.0, 0, 0, 0, False, True, False))
        self.calendar.add(Event(10.0, 0, 1, 1, True, True, True))
        self.calendar.add(Event(20.0, 0, 2, 2, True, False, False))
        self.calendar.add(Event(30.0, 0, 3, 3, True, True, True))
        self.current_event = self._next_decision_event()
        self.terminal_penalty_applied = False
        obs = self._build_obs_for_current_event()
        return obs, {"event": self.current_event}

    def get_action_mask(self) -> np.ndarray:
        return feasible_action_mask(self.state["available_chargers"], self.state["battery"], self.state["battery_max"], self.state["charge_rate"])

    def get_feasible_actions(self) -> list[int]:
        return feasible_actions(self.get_action_mask())

    def repair_action(self, action_index) -> int:
        return repair_action(action_index, self.get_action_mask())

    def step(self, action_index):
        selected_duration = action_index_to_duration(action_index)
        mask = self.get_action_mask()
        executed_action_index = action_index if mask[action_index] else self.repair_action(action_index)
        executed_duration = action_index_to_duration(executed_action_index)
        info = {
            "selected_action_index": action_index,
            "executed_action_index": executed_action_index,
            "selected_duration": selected_duration,
            "executed_duration": executed_duration,
            "action_repaired": executed_action_index != action_index,
        }

        self.state["battery"] = apply_charge(self.state["battery"], executed_duration, self.state["charge_rate"], self.state["battery_max"])
        self.state["available_chargers"] = occupy_charger(self.state["available_chargers"], executed_duration)

        is_ordinary = not self.current_event.integrated
        q_f = 0 if is_ordinary else min(2, self.state["onboard_parcels"])
        alighting = 2
        pax = simulate_passenger_stop(self.state["queue"], self.state["onboard_passengers"], 40, alighting, arrivals_during=1)
        self.state["onboard_passengers"] = pax["onboard_final"]
        self.state["queue"] = pax["queue_final"]

        self.state["onboard_parcels"], unloaded = unload_parcels(self.state["onboard_parcels"], q_f)
        self.state["locker"] += unloaded

        dwell = compute_dwell_breakdown(alighting, pax["initial_board"], 1.0, 1.0, q_f, executed_duration, 0.5, 0.5, 1.0, self.state["onboard_passengers"], True)
        self.state["time"] = advance_time(self.current_event.time, dwell.t_s, self.state["travel_time"])
        self.state["battery"] = apply_travel_consumption(self.state["battery"], self.state["travel_consumption"])
        self.state["trip_location"] += 1
        self.state["available_chargers"] = release_charger(self.state["available_chargers"], self.state["total_chargers"])

        self.current_event = self._next_decision_event()
        has_future = self.current_event is not None
        terminated, reason = check_termination(self.state, has_future)
        penalty = 0.0
        if terminated and not self.terminal_penalty_applied and self.state["onboard_parcels"] > 0:
            penalty = float(self.state["onboard_parcels"])
            self.terminal_penalty_applied = True
        reward = compute_reward(dwell.passenger_delay, executed_duration, penalty)
        info["termination_reason"] = reason
        next_obs = self._build_obs_for_current_event()
        return next_obs, float(reward), terminated, False, info

    def _next_decision_event(self):
        while len(self.calendar) > 0:
            e = self.calendar.pop_next()
            if e.is_decision and e.integrated and e.requires_stop:
                return e
            # ordinary stop/event still processed at coarse level by time advance placeholder
        return None

    def _build_obs_for_current_event(self):
        e = self.current_event
        local = {
            "station_id": -1 if e is None else e.station_id,
            "trip_id": self.state["trip_id"],
            "arriving_battery": self.state["battery"],
            "onboard_before_alight": self.state["onboard_passengers"],
            "onboard_parcels_before_unload": self.state["onboard_parcels"],
            "alighting": 2,
            "initial_board": min(self.state["queue"], 40 - self.state["onboard_passengers"]),
            "q_f": 0 if e is None or not e.integrated else min(2, self.state["onboard_parcels"]),
            "available_chargers": self.state["available_chargers"],
            "locker": self.state["locker"],
            "idle_drones": self.state["idle_drones"],
            "full_batteries": self.state["full_batteries"],
            "power_margin": self.state["power_margin"],
            "delivery_urgency": self.state["parcel_urgency"],
        }
        self.state["calendar_len"] = len(self.calendar)
        return build_observation(self.state, local)
