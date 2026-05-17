import math

from src.data_generation.network_generator import generate_network
from src.env.ebus_drone_env import EBusDroneEnv


def _cfg():
    return {
        "network": {"num_stops": 4, "stop_spacing_km": 1.0},
        "bus": {"nominal_speed_kmh": 30.0},
        "operation": {"nominal_line_time_min": 20.0, "return_time_min": 12.0, "layover_time_min": 8.0},
    }


def test_fleet_size_matches_cycle_time_rule_and_trip_timing_fields_present():
    config = _cfg()
    inst = {"num_scheduled_bus_trips": 10, "num_scheduled_trips": 10, "planned_headway_min": 7.0}
    net = generate_network(config, inst, seed=1)
    expected = math.ceil((20.0 + 12.0 + 8.0) / 7.0)
    assert net["physical_fleet_size"] == expected
    for trip in net["scheduled_bus_trips"]:
        assert "terminal_arrival_min" in trip
        assert "return_completion_min" in trip
        assert "layover_completion_min" in trip


def test_vehicle_circulation_is_temporally_feasible_per_bus():
    config = _cfg()
    inst = {"num_scheduled_bus_trips": 14, "num_scheduled_trips": 14, "planned_headway_min": 6.0}
    net = generate_network(config, inst, seed=1)
    by_bus = {vid: [] for vid in net["physical_buses"]}
    for trip in net["scheduled_bus_trips"]:
        vid = net["vehicle_circulation"][int(trip["trip_id"])]
        by_bus[vid].append(trip)
    for seq in by_bus.values():
        seq.sort(key=lambda t: float(t["departure_min"]))
        for i in range(len(seq) - 1):
            cur = seq[i]
            nxt = seq[i + 1]
            assert float(cur["layover_completion_min"]) <= float(nxt["departure_min"]) + 1e-9


def test_non_service_return_energy_consumption_is_applied():
    env = EBusDroneEnv(smoke_test=True)
    env.state["travel_distance_km"] = 2.0
    env.state["travel_energy_kwh_per_km"] = 1.5
    env.instance["network"]["return_time_min"] = 10.0
    env.instance["network"]["nominal_line_time_min"] = 20.0
    bus = {"battery_kwh": 100.0}
    env._apply_non_service_return(bus)
    # return distance = 2.0 * (10/20) = 1.0 km; energy = 1.5 kWh
    assert abs(bus["battery_kwh"] - 98.5) < 1e-9


def test_no_decision_event_generated_on_non_service_return_window():
    config = _cfg()
    inst = {"num_scheduled_bus_trips": 8, "num_scheduled_trips": 8, "planned_headway_min": 8.0}
    net = generate_network(config, inst, seed=1)
    by_bus = {vid: [] for vid in net["physical_buses"]}
    for trip in net["scheduled_bus_trips"]:
        vid = net["vehicle_circulation"][int(trip["trip_id"])]
        by_bus[vid].append(trip)
    for seq in by_bus.values():
        seq.sort(key=lambda t: float(t["departure_min"]))
        for i in range(len(seq) - 1):
            cur = seq[i]
            nxt = seq[i + 1]
            # no passenger/parcel/charging/drone decisions occur on return; next decision opportunity cannot predate layover completion
            assert float(nxt["departure_min"]) >= float(cur["layover_completion_min"]) - 1e-9
