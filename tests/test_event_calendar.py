from src.env.event_calendar import EventCalendar


def test_event_calendar_from_generated_data_non_decreasing():
    instance = {
        "network": {"stops": [{"stop_id": 1}, {"stop_id": 2}, {"stop_id": 3}], "nominal_travel_time_min": [[0, 5, 10], [5, 0, 5], [10, 5, 0]], "scheduled_bus_trips": [{"trip_id": 1, "departure_min": 0}]},
        "stations": {"stations": [{"station_id": 2, "stop_id": 2}]},
    }
    scenario = {"passenger": {"passenger_arrivals": {1: [0, 1], 2: [1], 3: [0]}}}
    assignment = {"decisions": [{"trip_id": 1, "station_id": 2, "customer_id": 1}]}
    c = EventCalendar(); c.build_from_generated(instance, scenario, assignment)
    times = []
    integrated_seen = ordinary_seen = False
    while len(c) > 0:
        e = c.pop_next(); times.append(e.time)
        integrated_seen = integrated_seen or e.integrated
        ordinary_seen = ordinary_seen or (not e.integrated)
    assert len(times) > 0
    assert integrated_seen
    assert ordinary_seen
    assert times == sorted(times)
