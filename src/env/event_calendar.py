from __future__ import annotations

import heapq
from typing import Optional

from .event_types import Event


class EventCalendar:
    def __init__(self) -> None:
        self._heap: list[Event] = []

    def add(self, event: Event) -> None:
        heapq.heappush(self._heap, event)

    def add_bus_arrival(self, *, time: float, trip_id: int, stop_index: int, station_id: int, integrated: bool, passengers_required: bool, parcel_required: bool) -> None:
        requires_stop = bool(passengers_required or parcel_required)
        self.add(Event(time=float(time), kind="bus_arrival", trip_id=int(trip_id), stop_index=int(stop_index), station_id=int(station_id), integrated=bool(integrated), requires_stop=requires_stop, is_decision=bool(integrated and requires_stop), passengers_required=bool(passengers_required), parcel_required=bool(parcel_required)))

    def add_charger_release(self, *, time: float, station_id: int) -> None:
        self.add(Event(time=float(time), kind="charger_release", station_id=int(station_id)))

    def add_station_tick(self, *, time: float, station_id: int) -> None:
        self.add(Event(time=float(time), kind="dispatch_tick", station_id=int(station_id)))

    def build_from_generated(self, instance: dict, scenario: dict, assignment: dict) -> None:
        stations = {int(s.get("stop_id", s["station_id"])): int(s["station_id"]) for s in instance["stations"]["stations"]}
        travel = instance["network"]["nominal_travel_time_min"]
        trips = instance["network"]["scheduled_bus_trips"]
        stop_ids = [int(s["stop_id"]) for s in instance["network"]["stops"]]
        assigned = {(int(d["trip_id"]), int(d["station_id"])) for d in assignment.get("decisions", [])}
        arrivals = scenario.get("passenger", {}).get("passenger_arrivals", {})
        for trip in trips:
            t_id = int(trip["trip_id"])
            dep = float(trip["departure_min"])
            for idx, stop_id in enumerate(stop_ids):
                arr_time = dep + float(travel[0][idx])
                integrated = stop_id in stations
                station_id = stations.get(stop_id, -1)
                pax_required = bool(arrivals.get(str(stop_id), arrivals.get(stop_id, [])))
                parcel_required = integrated and ((t_id, station_id) in assigned)
                self.add_bus_arrival(time=arr_time, trip_id=t_id, stop_index=idx, station_id=station_id, integrated=integrated, passengers_required=pax_required, parcel_required=parcel_required)

    def pop_next(self) -> Optional[Event]:
        return heapq.heappop(self._heap) if self._heap else None

    def peek_next(self) -> Optional[Event]:
        return self._heap[0] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)
