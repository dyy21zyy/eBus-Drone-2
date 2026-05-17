from __future__ import annotations
from src.data_generation.network_generator import generate_network
from src.data_generation.station_generator import generate_stations
from src.data_generation.parcel_generator import generate_customers_and_parcels
from src.data_generation.passenger_generator import generate_passenger_parameters, generate_passenger_scenario
from src.data_generation.power_load_generator import generate_station_base_load

def _resolve_horizons(config: dict) -> tuple[float, float, float]:
    gen = config.get("generation", {})
    legacy = float(gen.get("horizon_minutes", 480))
    t_bus = float(gen.get("bus_operation_horizon_minutes", legacy))
    t_del = float(gen.get("delivery_evaluation_horizon_minutes", legacy))
    if t_del < t_bus:
        raise ValueError(f"delivery_evaluation_horizon_minutes (T_del={t_del}) must be >= bus_operation_horizon_minutes (T_bus={t_bus})")
    return t_bus, t_del, t_del


def _validate_generation_inputs(config: dict, instance_cfg: dict):
    num_scheduled = int(instance_cfg.get("num_scheduled_trips", instance_cfg["num_scheduled_bus_trips"]))
    num_freight = int(instance_cfg.get("num_freight_carrying_trips", num_scheduled))
    if num_freight > num_scheduled:
        raise ValueError(f"num_freight_carrying_trips ({num_freight}) must be <= num_scheduled_trips ({num_scheduled})")
    if float(config["parcel"].get("drone_service_radius_km", 0.0)) <= 0.0:
        raise ValueError("drone_service_radius_km must be > 0")
    if float(config["charging"].get("pantograph_power_kw", 0.0)) <= 0.0:
        raise ValueError("pantograph_power_kw must be positive")
    if float(config["power"].get("station_capacity_kw", 0.0)) <= 0.0:
        raise ValueError("station_capacity_kw must be positive")
    if float(config["battery"].get("charge_duration_min", 0.0)) <= 0.0:
        raise ValueError("battery.charge_duration_min must be positive")


def generate_instance(config: dict, instance_cfg: dict, seed: int) -> dict:
    _validate_generation_inputs(config, instance_cfg)
    t_bus, t_del, horizon = _resolve_horizons(config)
    network = generate_network(config, instance_cfg, seed=seed)
    stations = generate_stations(config, instance_cfg)
    customers = generate_customers_and_parcels(
        config,
        instance_cfg,
        network["stops"],
        stations["station_ids"],
        network["scheduled_bus_trips"],
        network["freight_carrying_trip_ids"],
        network["nominal_travel_time_min"],
        seed,
    )
    deadline_repaired = False
    passenger = generate_passenger_parameters(config, network["stops"], seed)
    return {
        "instance_name": instance_cfg["name"],
        "seed": seed,
        "bus_operation_horizon_minutes": t_bus,
        "delivery_evaluation_horizon_minutes": t_del,
        "horizon_minutes": horizon,
        "network": network,
        "stations": stations,
        "customers": customers["customers"],
        "passenger": passenger,
        "bus": config["bus"],
        "charging": config["charging"],
        "drone": config["drone"],
        "battery": config["battery"],
        "parcel": config["parcel"],
        "power": config["power"],
        "generation_metadata": {
            "deadline_repair_policy": "paper_slack_from_earliest_planned_completion",
            "deadline_repaired": deadline_repaired,
            "num_scheduled_trips": int(network["num_scheduled_trips"]),
            "num_freight_carrying_trips": int(network["num_freight_carrying_trips"]),
            "freight_carrying_trip_ids": [int(tid) for tid in network["freight_carrying_trip_ids"]],
            "drone_service_radius_km": float(config["parcel"].get("drone_service_radius_km", config["parcel"].get("drone_round_trip_range_km", 8.0))),
            "battery_charge_duration_min": float(config["battery"].get("charge_duration_min", 45.0)),
            "max_simultaneous_battery_charging": int(config["battery"].get("max_simultaneous_charging", 6)),
        },
    }


def generate_scenario(config: dict, instance: dict, seed: int, scenario_index: int = 0) -> dict:
    sc_seed = seed * 1000 + scenario_index
    stops = instance["network"]["stops"]
    station_ids = instance["stations"]["station_ids"]
    return {
        "scenario_index": scenario_index,
        "seed": sc_seed,
        "passenger": generate_passenger_scenario(config, stops, sc_seed),
        "power": generate_station_base_load(config, station_ids, sc_seed),
    }
