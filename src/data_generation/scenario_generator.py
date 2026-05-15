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
        t_del = t_bus
    return t_bus, t_del, t_del


def generate_instance(config: dict, instance_cfg: dict, seed: int) -> dict:
    t_bus, t_del, horizon = _resolve_horizons(config)
    network = generate_network(config, instance_cfg, seed=seed)
    stations = generate_stations(config, instance_cfg)
    customers = generate_customers_and_parcels(config, instance_cfg, network["stops"], stations["station_ids"], seed)
    wait_nominal = float(config["parcel"]["nominal_locker_waiting_time_min"])
    nominal_unloading = float(config["parcel"].get("nominal_unloading_time_min", (config["parcel"]["unloading_capacity_kg_per_stop"] * config["parcel"]["unloading_time_sec_per_kg"]) / 60.0))
    min_departure = min(float(t["departure_min"]) for t in network["scheduled_bus_trips"])
    travel = network["nominal_travel_time_min"][0]
    deadline_repaired = False
    for c in customers["customers"]:
        min_completion = float("inf")
        for opt in c["feasible_stations"]:
            sid = int(opt["station_id"])
            outbound = float(opt["mission_duration_min"]) * 0.5
            arrival = min_departure + float(travel[sid - 1])
            completion = arrival + nominal_unloading + wait_nominal + outbound
            min_completion = min(min_completion, completion)
        if c["delivery_deadline_min"] < min_completion:
            c["delivery_deadline_min"] = round(min_completion, 4)
            deadline_repaired = True
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
            "deadline_repair_policy": "repair_to_min_completion",
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
