from __future__ import annotations

import pytest

from src.utils.config import load_yaml, validate_config


def test_default_config_matches_paper_defaults():
    cfg = load_yaml("configs/default.yaml")

    assert cfg["network"]["num_stops"] == 30
    assert cfg["network"]["interstop_distance_km"] == 1.0
    assert cfg["generation"]["bus_operation_horizon_minutes"] == 360
    assert cfg["generation"]["delivery_evaluation_horizon_minutes"] == 480

    assert cfg["bus"]["passenger_capacity"] == 80
    assert cfg["bus"]["battery_capacity_kwh"] == 160.0
    assert cfg["bus"]["safety_battery_kwh"] == 40.0
    assert cfg["bus"]["initial_battery_fraction_min"] == 0.55
    assert cfg["bus"]["initial_battery_fraction_max"] == 0.85
    assert cfg["bus"]["energy_kwh_per_km"] == 1.6
    assert cfg["bus"]["nominal_speed_kmh"] == 30.0

    assert cfg["charging"]["chargers_per_station"] == 2
    assert cfg["charging"]["pantograph_power_kw"] == 500.0
    assert cfg["charging"]["efficiency"] == 0.95
    assert cfg["charging"]["action_set_seconds"] == [0, 15, 30, 45, 60, 75, 90, 105, 120]
    assert cfg["charging"]["max_single_stop_seconds"] == 120

    assert cfg["passenger"]["baseline_rate_mean_per_min"] == 0.25
    assert cfg["passenger"]["baseline_rate_std_per_min"] == 0.10
    assert cfg["passenger"]["baseline_rate_min_per_min"] == 0.05
    assert cfg["passenger"]["baseline_rate_max_per_min"] == 0.60
    assert cfg["passenger"]["boarding_time_sec_per_passenger"] == 3.0
    assert cfg["passenger"]["alighting_time_sec_per_passenger"] == 1.5

    assert cfg["parcel"]["drone_service_radius_km"] == 8.0
    assert cfg["bus"]["freight_capacity_kg"] == 20.0
    assert cfg["parcel"]["unloading_capacity_kg_per_stop"] == 10.0
    assert cfg["parcel"]["locker_capacity_kg"] == 30.0
    assert cfg["parcel"]["unloading_time_sec_per_kg"] == 6.0
    assert cfg["parcel"]["nominal_locker_waiting_time_min"] == 10.0

    assert cfg["drone"]["drones_per_station"] == 3
    assert cfg["drone"]["speed_kmh"] == 40.0
    assert cfg["drone"]["max_round_trip_duration_min"] == 120.0
    assert cfg["drone"]["customer_service_time_min"] == 1.0
    assert cfg["drone"]["turnaround_time_min"] == 1.0

    assert cfg["battery"]["initial_fully_charged_per_station"] == 6
    assert cfg["battery"]["initial_depleted_per_station"] == 0
    assert cfg["battery"]["charge_power_kw"] == 2.0
    assert cfg["battery"]["charge_duration_min"] == 45.0
    assert cfg["battery"]["max_simultaneous_charging"] == 6
    assert cfg["battery"]["charging_mode"] == "non_preemptive"

    assert cfg["power"]["station_capacity_kw"] == 1100.0
    assert cfg["power"]["nominal_base_load_min_kw"] == 80.0
    assert cfg["power"]["nominal_base_load_max_kw"] == 180.0

    assert cfg["rl"]["episodes"] == 5000
    assert cfg["rl"]["evaluation_episodes"] == 100
    assert cfg["rl"]["random_seeds"] == 5
    assert cfg["rl"]["replay_buffer_size"] == 500000
    assert cfg["rl"]["batch_size"] == 128
    assert cfg["rl"]["optimizer"] == "Adam"
    assert cfg["rl"]["learning_rate"] == pytest.approx(5.0e-5)
    assert cfg["rl"]["gamma"] == pytest.approx(0.997)
    assert cfg["rl"]["epsilon_start"] == 1.0
    assert cfg["rl"]["epsilon_end"] == 0.05
    assert cfg["rl"]["epsilon_decay_fraction"] == pytest.approx(0.8)
    assert cfg["rl"]["polyak_tau"] == pytest.approx(0.005)
    assert cfg["rl"]["hidden_layers"] == [256, 256, 128]
    assert cfg["rl"]["activation"] == "ReLU"

    assert [cfg["reward"][f"alpha_{i}"] for i in range(1, 7)] == [1.0, 1.0, 0.2, 1.0, 5.0, 1.0]


def test_default_instance_profiles_match_paper():
    small = load_yaml("configs/instances/small.yaml")
    medium = load_yaml("configs/instances/medium.yaml")
    large = load_yaml("configs/instances/large.yaml")

    assert small["num_customers"] == 30
    assert len(small["station_ids"]) == 6
    assert small["num_scheduled_trips"] == 24
    assert small["num_freight_carrying_trips"] == 8
    assert small["planned_headway_min"] == 15

    assert medium["num_customers"] == 60
    assert len(medium["station_ids"]) == 8
    assert medium["num_scheduled_trips"] == 36
    assert medium["num_freight_carrying_trips"] == 12
    assert medium["planned_headway_min"] == 10

    assert large["num_customers"] == 90
    assert len(large["station_ids"]) == 8
    assert large["num_scheduled_trips"] == 45
    assert large["num_freight_carrying_trips"] == 16
    assert large["planned_headway_min"] == 8


def test_invalid_config_validation_errors_are_clear():
    cfg = load_yaml("configs/default.yaml")

    bad_action_set = dict(cfg)
    bad_action_set["charging"] = dict(cfg["charging"])
    bad_action_set["charging"]["action_set_seconds"] = [15, 30]
    with pytest.raises(ValueError, match="must include 0"):
        validate_config(bad_action_set)

    bad_umax = dict(cfg)
    bad_umax["charging"] = dict(cfg["charging"])
    bad_umax["charging"]["max_single_stop_seconds"] = 90
    with pytest.raises(ValueError, match="must equal max"):
        validate_config(bad_umax)

    bad_horizon = dict(cfg)
    bad_horizon["generation"] = dict(cfg["generation"])
    bad_horizon["generation"]["delivery_evaluation_horizon_minutes"] = 100
    with pytest.raises(ValueError, match="must be >="):
        validate_config(bad_horizon)

    bad_power = dict(cfg)
    bad_power["power"] = dict(cfg["power"])
    bad_power["power"]["station_capacity_kw"] = 0
    with pytest.raises(ValueError, match="must be > 0"):
        validate_config(bad_power)

    bad_safety = dict(cfg)
    bad_safety["bus"] = dict(cfg["bus"])
    bad_safety["bus"]["safety_battery_kwh"] = bad_safety["bus"]["battery_capacity_kwh"]
    with pytest.raises(ValueError, match="must be <"):
        validate_config(bad_safety)

    bad_initial = dict(cfg)
    bad_initial["bus"] = dict(cfg["bus"])
    bad_initial["bus"]["initial_battery_fraction_max"] = 1.1
    with pytest.raises(ValueError, match="initial battery range"):
        validate_config(bad_initial)
