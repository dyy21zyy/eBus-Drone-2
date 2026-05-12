import argparse
from src.utils.config import load_yaml
from src.utils.logger import log_line
from src.data_generation.network_generator import generate_corridor_network
from src.data_generation.station_generator import select_integrated_stations
from src.data_generation.parcel_generator import generate_customers
from src.data_generation.instance_writer import write_instance_and_scenarios
from src.data_generation.scenario_generator import generate_scenario


def _build_instance(cfg: dict, instance_cfg: dict, seed: int) -> dict:
    import random
    rng = random.Random(seed)
    net = generate_corridor_network(cfg["network"]["num_stops"], cfg["network"]["stop_spacing_km"], cfg["network"]["horizon_minutes"])
    stations = select_integrated_stations(cfg["network"]["integrated_stations_medium"], instance_cfg["num_integrated_stations"])
    customers = generate_customers(instance_cfg["num_customers"], [s["position_km"] for s in net["stops"]], stations, cfg, rng)
    headway = instance_cfg["planned_headway_min"]
    trips = [{"trip_id": i+1, "departure_min": i*headway} for i in range(instance_cfg["num_scheduled_bus_trips"])]
    return {
        "instance_name": instance_cfg["name"],
        "seed": seed,
        "network": net,
        "stations": stations,
        "bus_trips": trips,
        "customers": customers,
        "bus": cfg["bus"],
        "charging": cfg["charging"],
        "offline_assignment": cfg["offline_assignment"],
        "drone": cfg["drone"],
        "battery": cfg["battery"],
        "station_power": cfg["station_power"],
    }


def run_generate(cfg, instance, seed):
    icfg = load_yaml(f"configs/instances/{instance}.yaml")
    inst = _build_instance(cfg, icfg, seed)
    scenarios = [generate_scenario(inst, cfg, seed + i) for i in range(cfg.get("scenario_count", 1))]
    out_dir = f"data/generated/{instance}"
    files = write_instance_and_scenarios(inst, scenarios, out_dir)
    log_line("outputs/logs/generate.log", f"generated {files}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['generate','offline','train','eval','benchmark','ablation','sensitivity','all'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--instance', default='small')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.mode in ['generate', 'all']:
        run_generate(cfg, args.instance, args.seed)
    else:
        log_line("outputs/logs/pipeline.log", f"mode {args.mode} invoked")


if __name__ == '__main__':
    main()
