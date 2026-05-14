from __future__ import annotations

from pathlib import Path
import csv
import json
import time

from src.rl.agents.dqn_dr_agent import DQNDRAgent
from src.rl.agents.ddqn_dr_agent import DDQNDRAgent
from src.rl.agents.am_ddqn_dr_agent import AMDDQNDRAgent
from src.rl.agents.am_dueling_ddqn_dr_agent import AMDuelingDDQNDRAgent

LEARNING_METHODS = {"dqn_dr", "ddqn_dr", "am_ddqn_dr", "proposed", "am_dueling_ddqn_dr"}


def _agent_cls(method: str):
    return {
        "dqn_dr": DQNDRAgent,
        "ddqn_dr": DDQNDRAgent,
        "am_ddqn_dr": AMDDQNDRAgent,
        "proposed": AMDuelingDDQNDRAgent,
        "am_dueling_ddqn_dr": AMDuelingDDQNDRAgent,
    }.get(method, AMDuelingDDQNDRAgent)


def train_agent(env, method: str = "proposed", episodes: int = 5, max_steps: int | None = 100, smoke_test: bool = False, out_root: str = "outputs", cfg: dict | None = None, seed: int = 0, instance_name: str = "unknown"):
    obs, _ = env.reset(seed=seed)
    cfg = dict(cfg or {})
    rl_cfg = dict(cfg.get("rl", {}))
    rl_cfg.setdefault("batch_size", 8 if smoke_test else 32)
    rl_cfg.setdefault("warmup_steps", 4 if smoke_test else 20)
    rl_cfg.setdefault("target_update_interval", 10)
    rl_cfg.setdefault("replay_buffer_size", 2000)
    rl_cfg.setdefault("action_set_seconds", cfg.get("charging", {}).get("action_set_seconds", []))
    cls = _agent_cls(method)
    agent = cls(len(obs), len(env.get_action_mask()), rl_cfg)
    episodes = int(rl_cfg.get("episodes", episodes))
    allow_train_truncation = bool(rl_cfg.get("allow_train_truncation", False))
    cfg_max_steps = rl_cfg.get("max_steps_per_episode", max_steps)
    if smoke_test or allow_train_truncation:
        max_steps = int(cfg_max_steps) if cfg_max_steps is not None else None
    else:
        max_steps = None
    rows = []
    t0 = time.time()
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_cost = 0.0
        losses = []
        bat_dep = 0
        dec = 0
        step_idx = 0
        termination_reason = None
        truncated_by_max_steps = False
        while True:
            if max_steps is not None and step_idx >= max_steps:
                truncated_by_max_steps = True
                termination_reason = "max_steps_truncated"
                break
            mask = env.get_action_mask(); action = agent.select_action(obs, mask, training=True)
            nxt, reward, term, trunc, info = env.step(action)
            nxt_mask = env.get_action_mask() if not (term or trunc) else mask
            agent.observe(obs, action, reward, nxt, term or trunc, mask, nxt_mask, info)
            loss = agent.update(); obs = nxt; ep_reward += reward; dec += 1
            step_idx += 1
            if loss is not None: losses.append(float(loss))
            rc = info.get("reward_components", {})
            ep_cost += rc.get("total_cost", -reward)
            bat_dep += int(rc.get("battery_safety", 0) > 0)
            if term or trunc:
                termination_reason = info.get("termination_reason") or ("truncated" if trunc else None)
                break
        operating_horizon = float(getattr(env, "horizon_sec", 0.0)) / 60.0
        episode_end_time = float(getattr(env, "state", {}).get("time", 0.0)) / 60.0
        rows.append({
            "episode": ep,
            "episode_reward": ep_reward,
            "episode_cost": ep_cost,
            "epsilon": agent._eps(),
            "loss": sum(losses)/len(losses) if losses else "",
            "number_decision_events": dec,
            "battery_depletion_count": bat_dep,
            "episode_steps": dec,
            "episode_end_time": episode_end_time,
            "operating_horizon_min": operating_horizon,
            "termination_reason": termination_reason or ("horizon_reached" if episode_end_time >= operating_horizon else "unknown"),
            "full_horizon_completed": bool((termination_reason == "horizon_reached") and not truncated_by_max_steps),
            "truncated_by_max_steps": bool(truncated_by_max_steps),
            "paper_ready_episode": bool((not smoke_test) and (not truncated_by_max_steps)),
        })
    out = Path(out_root)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "raw_logs").mkdir(parents=True, exist_ok=True)
    (out / "configs").mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoints" / f"{method}_{instance_name}_seed_{seed}.pt"
    agent.save_checkpoint(str(ckpt))
    (out / "checkpoints" / f"{method}.pt").write_bytes(ckpt.read_bytes())
    csv_path = out / "metrics" / f"train_log_{method}_{instance_name}_seed_{seed}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with (out / "raw_logs" / f"training_curve_{method}_{instance_name}_seed_{seed}.json").open("w", encoding="utf-8") as f:
        json.dump({"seed": seed, "method": method, "instance": instance_name, "runtime_sec": time.time()-t0, "rows": rows}, f, indent=2)
    with (out / "configs" / f"train_{method}_{instance_name}_seed_{seed}.json").open("w", encoding="utf-8") as f:
        json.dump({"seed": seed, "method": method, "instance": instance_name, "config": cfg}, f, indent=2)
    return agent, str(ckpt)
