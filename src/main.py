import argparse
from src.utils.config import load_yaml
from src.data_generation.scenario_generator import generate_instance, generate_scenario
from src.data_generation.instance_writer import write_instance


def run_generate(cfg, instance_name: str, seed: int):
    instance_cfg = load_yaml(f"configs/instances/{instance_name}.yaml")
    out_root = f"{cfg['paths']['data_generated']}/{instance_name}"
    instance = generate_instance(cfg, instance_cfg, seed)
    write_instance(instance, out_root, f"instance_seed_{seed}")
    num_scen = int(cfg["generation"].get("num_scenarios", 1))
    for idx in range(num_scen):
        scenario = generate_scenario(cfg, instance, seed, idx)
        write_instance(scenario, out_root, f"scenario_{idx}_seed_{seed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['generate'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--instance', default='small')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    run_generate(cfg, args.instance, args.seed)


if __name__ == '__main__':
    main()
