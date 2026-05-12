from __future__ import annotations
from src.data_generation.network_generator import generate_network
from src.data_generation.station_generator import generate_stations
from src.data_generation.parcel_generator import generate_customers_and_parcels
from src.data_generation.passenger_generator import generate_passenger_parameters, generate_passenger_scenario
from src.data_generation.power_load_generator import generate_station_base_load


def generate_instance(config: dict, instance_cfg: dict, seed: int) -> dict:
    network = generate_network(config, instance_cfg)
    stations = generate_stations(config, instance_cfg)
    customers = generate_customers_and_parcels(config, instance_cfg, network["stops"], stations["station_ids"], seed)
    passenger = generate_passenger_parameters(config, network["stops"], seed)
    return {
        "instance_name": instance_cfg["name"],
        "seed": seed,
        "horizon_minutes": config["generation"]["horizon_minutes"],
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
