from __future__ import annotations

import heapq
from typing import Optional

from .event_types import Event


class EventCalendar:
    def __init__(self) -> None:
        self._heap: list[Event] = []

    def add(self, event: Event) -> None:
        heapq.heappush(self._heap, event)

    def build_from_generated(self, instance: dict, scenario: dict, assignment: dict) -> None:
        stations = {int(s["station_id"]): int(s["station_id"]) for s in instance["stations"]["stations"]}
        travel = instance["network"]["nominal_travel_time_min"]
        trips = instance["network"]["scheduled_bus_trips"]
        stop_ids = [int(s["stop_id"]) for s in instance["network"]["stops"]]
        decisions = assignment.get("decisions", [])
        assigned = {(int(d["trip_id"]), int(d["station_id"])) for d in decisions}
        arrivals = scenario.get("passenger", {}).get("passenger_arrivals", {})

        for trip in trips:
            t_id = int(trip["trip_id"])
            dep = float(trip["departure_min"])
            for idx, stop_id in enumerate(stop_ids):
                arr_time = dep + float(travel[0][idx])
                integrated = stop_id in stations
                station_id = stations.get(stop_id, -1)
                pax_required = str(stop_id) in arrivals and bool(arrivals.get(str(stop_id)))
                parcel_required = integrated and ((t_id, station_id) in assigned)
                is_decision = integrated and (pax_required or parcel_required)
                self.add(Event(arr_time, t_id, idx, station_id, integrated, True, is_decision, pax_required, parcel_required))

    def pop_next(self) -> Optional[Event]:
        if not self._heap:
            return None
        return heapq.heappop(self._heap)

    def peek_next(self) -> Optional[Event]:
        if not self._heap:
            return None
        return self._heap[0]

    def __len__(self) -> int:
        return len(self._heap)
