import json
from pathlib import Path
from src.utils.config import load_yaml
from src.data_generation.scenario_generator import generate_instance, generate_scenario
from src.offline.assignment_data_builder import build_assignment_data
from src.main import run_generate
from src.data_generation.parcel_generator import generate_customers_and_parcels


def test_instance_constraints():
    cfg = load_yaml('configs/default.yaml')
    inst_cfg = load_yaml('configs/instances/small.yaml')
    instance = generate_instance(cfg, inst_cfg, seed=1)
    assert instance['network']['scheduled_bus_trips']
    assert instance['network']['stops']
    assert instance['stations']['stations']
    assert instance['customers']
    assert instance['network']['nominal_travel_time_min']
    assert instance["bus_operation_horizon_minutes"] == 360
    assert instance["delivery_evaluation_horizon_minutes"] == 480
    assert instance["horizon_minutes"] == 480
    assert instance["network"]["num_scheduled_trips"] == 24
    assert instance["network"]["num_freight_carrying_trips"] == 8
    assert len(instance["network"]["freight_carrying_trip_ids"]) == 8
    assert instance["bus"]["passenger_capacity"] == 80
    assert instance["charging"]["pantograph_power_kw"] == 500.0
    assert instance["power"]["station_capacity_kw"] == 1100.0
    assert instance["parcel"]["locker_capacity_kg"] == 30.0


def test_default_and_instance_configs_match_paper_settings():
    cfg = load_yaml('configs/default.yaml')
    medium = load_yaml('configs/instances/medium.yaml')
    assert cfg["generation"]["bus_operation_horizon_minutes"] == 360
    assert cfg["generation"]["delivery_evaluation_horizon_minutes"] == 480
    assert cfg["network"]["num_stops"] == 30
    assert cfg["charging"]["chargers_per_station"] == 2
    assert cfg["battery"]["charge_duration_min"] == 45.0
    assert cfg["rl"]["episodes"] == 5000
    assert medium["station_ids"] == [1, 5, 9, 13, 17, 21, 25, 29]
    assert medium["planned_headway_min"] == 10
    assert medium["num_scheduled_trips"] == 36
    assert medium["num_freight_carrying_trips"] == 12


def test_generate_command_outputs():
    cfg = load_yaml('configs/default.yaml')
    run_generate(cfg, "small", 1)
    out = Path('data/generated/small')
    assert (out / 'instance_seed_1.json').exists()
    assert (out / 'scenario_0_seed_1.json').exists()
    sc = json.loads((out / 'scenario_0_seed_1.json').read_text())
    assert 'passenger' in sc and 'power' in sc


def test_generated_deadlines_have_planned_feasible_pair():
    cfg = load_yaml('configs/default.yaml')
    inst_cfg = load_yaml('configs/instances/small.yaml')
    instance = generate_instance(cfg, inst_cfg, seed=2)
    data = build_assignment_data(instance)
    for i in data.customers:
        feasible = False
        for h in data.feasible_stations_by_customer[i]:
            for b in data.trips:
                key = (b, h, i)
                if data.c_bhi_0[key] <= data.deadline[i] + 1e-9:
                    feasible = True
                    break
            if feasible:
                break
        assert feasible, f"customer {i} has no planned-feasible (trip, station) pair"


def test_customer_generation_matches_paper_rules():
    cfg = load_yaml('configs/default.yaml')
    inst_cfg = load_yaml('configs/instances/medium.yaml')
    instance = generate_instance(cfg, inst_cfg, seed=7)
    allowed_weights = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5}
    for c in instance["customers"]:
        assert c["parcel_weight_kg"] in allowed_weights
        assert c["feasible_station_ids"]
        assert sorted(c["feasible_station_ids"]) == sorted(int(o["station_id"]) for o in c["feasible_stations"])
        assert c["deadline_class"] in {"tight", "moderate", "loose"}
        assert c["delivery_deadline_min"] >= c["earliest_planned_completion_min"]


def test_deadline_class_mix_approximation_large_sample():
    cfg = load_yaml('configs/default.yaml')
    inst_cfg = load_yaml('configs/instances/large.yaml')
    counts = {"tight": 0, "moderate": 0, "loose": 0}
    total = 0
    for seed in range(10, 30):
        instance = generate_instance(cfg, inst_cfg, seed=seed)
        for c in instance["customers"]:
            counts[c["deadline_class"]] += 1
            total += 1
    mix = {k: counts[k] / total for k in counts}
    assert abs(mix["tight"] - 0.30) <= 0.06
    assert abs(mix["moderate"] - 0.50) <= 0.08
    assert abs(mix["loose"] - 0.20) <= 0.06


def test_drone_service_radius_reachability_threshold():
    config = {
        "network": {"service_area": {"x_min_km": 0.0, "x_max_km": 0.0, "y_min_km": 7.9, "y_max_km": 7.9}},
        "parcel": {
            "weight_values_kg": [1.0],
            "drone_service_radius_km": 8.0,
            "nominal_locker_waiting_time_min": 10.0,
            "nominal_unloading_time_min": 0.0,
        },
        "drone": {"speed_kmh": 40.0, "customer_service_time_min": 1.0, "turnaround_time_min": 1.0, "max_round_trip_duration_min": 120.0},
        "generation": {"bus_operation_horizon_minutes": 360.0, "max_customer_generation_retries": 5},
    }
    instance_cfg = {"num_customers": 1}
    stops = [{"stop_id": 1, "x_km": 0.0}, {"stop_id": 2, "x_km": 0.0}]
    trips = [{"trip_id": 1, "departure_min": 0.0}]
    travel = [[0.0, 0.0], [0.0, 0.0]]
    data = generate_customers_and_parcels(config, instance_cfg, stops, [1, 2], trips, [1], travel, seed=1)
    d = float(data["customers"][0]["feasible_stations"][0]["distance_km"])
    assert d <= 8.0

    config["network"]["service_area"]["y_min_km"] = 8.1
    config["network"]["service_area"]["y_max_km"] = 8.1
    try:
        generate_customers_and_parcels(config, instance_cfg, stops, [1], trips, [1], travel, seed=1)
        assert False, "expected generation failure for unreachable customer"
    except ValueError:
        assert True


def test_generation_validation_guards():
    cfg = load_yaml('configs/default.yaml')
    inst_cfg = load_yaml('configs/instances/small.yaml')

    bad_horizon = dict(cfg)
    bad_horizon["generation"] = dict(cfg["generation"])
    bad_horizon["generation"]["delivery_evaluation_horizon_minutes"] = 300
    try:
        generate_instance(bad_horizon, inst_cfg, seed=1)
        assert False, "Expected invalid horizon validation failure"
    except ValueError as e:
        assert "must be >=" in str(e)

    bad_freq = dict(inst_cfg)
    bad_freq["num_freight_carrying_trips"] = bad_freq["num_scheduled_trips"] + 1
    try:
        generate_instance(cfg, bad_freq, seed=1)
        assert False, "Expected invalid freight trips validation failure"
    except ValueError as e:
        assert "<=" in str(e)
