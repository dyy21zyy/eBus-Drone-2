import argparse
import json
from pathlib import Path

from src.utils.config import load_yaml
from src.data_generation.scenario_generator import generate_instance, generate_scenario
from src.data_generation.instance_writer import write_instance
from src.offline.assignment_data_builder import build_assignment_data
from src.offline.assignment_io import write_assignment
from src.offline.assignment_solver import solve_assignment
from src.env.ebus_drone_env import EBusDroneEnv
from src.policies.no_charging_policy import NoChargingPolicy
from src.policies.max_feasible_policy import MaxFeasiblePolicy


def run_generate(cfg, instance_name: str, seed: int):
    instance_cfg = load_yaml(f"configs/instances/{instance_name}.yaml")
    out_root = f"{cfg['paths']['data_generated']}/{instance_name}"
    instance = generate_instance(cfg, instance_cfg, seed)
    write_instance(instance, out_root, f"instance_seed_{seed}")
    num_scen = int(cfg["generation"].get("num_scenarios", 1))
    for idx in range(num_scen):
        scenario = generate_scenario(cfg, instance, seed, idx)
        write_instance(scenario, out_root, f"scenario_{idx}_seed_{seed}")


def run_offline(cfg, instance_name: str, seed: int):
    instance_path = Path(cfg["paths"]["data_generated"]) / instance_name / f"instance_seed_{seed}.json"
    if not instance_path.exists():
        raise FileNotFoundError(f"Instance file not found: {instance_path}. Run generate mode first.")
    instance = json.loads(instance_path.read_text(encoding="utf-8"))
    data = build_assignment_data(instance)
    result = solve_assignment(data)
    out = Path(cfg["paths"]["outputs"]) / "assignments" / f"offline_assignment_{instance_name}_seed_{seed}.json"
    write_assignment(result, str(out))
    print(f"Saved offline assignment to {out}")




def run_eval(cfg, instance_name: str, seed: int, method: str, smoke_test: bool = False):
    _ = cfg, instance_name, seed
    env = EBusDroneEnv()
    obs, _ = env.reset(seed=seed)
    policy = NoChargingPolicy() if method == "no_charging" else MaxFeasiblePolicy()
    total_reward = 0.0
    max_steps = 3 if smoke_test else 50
    for _ in range(max_steps):
        mask = env.get_action_mask()
        action = policy.act(obs, mask)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"Eval finished method={method} total_reward={total_reward:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['generate', 'offline','eval'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--instance', default='small')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--method', default='no_charging')
    ap.add_argument('--smoke-test', action='store_true')
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.mode == "generate":
        run_generate(cfg, args.instance, args.seed)
    elif args.mode == "offline":
        run_offline(cfg, args.instance, args.seed)
    elif args.mode == "eval":
        run_eval(cfg, args.instance, args.seed, args.method, args.smoke_test)


if __name__ == '__main__':
    main()
