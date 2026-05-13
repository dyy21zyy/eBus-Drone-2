from __future__ import annotations

from pathlib import Path
import csv

from src.rl.agents.dqn_dr_agent import DQNDRAgent
from src.rl.agents.ddqn_dr_agent import DDQNDRAgent
from src.rl.agents.am_ddqn_dr_agent import AMDDQNDRAgent
from src.rl.agents.am_dueling_ddqn_dr_agent import AMDuelingDDQNDRAgent


def train_agent(env, method: str = "proposed", episodes: int = 5, max_steps: int = 100, smoke_test: bool = False, out_root: str = "outputs", cfg: dict | None = None):
    obs, _ = env.reset(seed=0)
    cfg = dict(cfg or {})
    rl_cfg = dict(cfg.get("rl", {}))
    rl_cfg.setdefault("batch_size", 8 if smoke_test else 32)
    rl_cfg.setdefault("warmup_steps", 4 if smoke_test else 20)
    rl_cfg.setdefault("target_update_interval", 10)
    rl_cfg.setdefault("replay_buffer_size", 2000)
    rl_cfg.setdefault("action_set_seconds", cfg.get("charging", {}).get("action_set_seconds", []))
    cls = {"dqn_dr": DQNDRAgent, "ddqn_dr": DDQNDRAgent, "am_ddqn_dr": AMDDQNDRAgent, "proposed": AMDuelingDDQNDRAgent, "am_dueling_ddqn_dr": AMDuelingDDQNDRAgent}.get(method, AMDuelingDDQNDRAgent)
    agent = cls(len(obs), len(env.get_action_mask()), rl_cfg)
    episodes = int(rl_cfg.get("episodes", episodes))
    max_steps = int(rl_cfg.get("max_steps_per_episode", max_steps))
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep); ep_reward = 0.0; ep_cost = 0.0; losses = []; bat_dep = 0; dec = 0
        for _ in range(max_steps):
            mask = env.get_action_mask(); action = agent.select_action(obs, mask, training=True)
            nxt, reward, term, trunc, info = env.step(action)
            nxt_mask = env.get_action_mask() if not (term or trunc) else mask
            agent.observe(obs, action, reward, nxt, term or trunc, mask, nxt_mask, info)
            loss = agent.update(); obs = nxt; ep_reward += reward; dec += 1
            if loss is not None: losses.append(loss)
            rc = info.get("reward_components", {})
            ep_cost += rc.get("total_cost", -reward)
            bat_dep += int(rc.get("battery_safety", 0) > 0)
            if term or trunc: break
        rows.append({"episode": ep, "episode_reward": ep_reward, "episode_cost": ep_cost, "epsilon": agent._eps(), "loss": sum(losses)/len(losses) if losses else "", "number_decision_events": dec, "battery_depletion_count": bat_dep})
    out = Path(out_root); (out / "checkpoints").mkdir(parents=True, exist_ok=True); (out / "metrics").mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoints" / f"{method}.pt"; agent.save_checkpoint(str(ckpt))
    csv_path = out / "metrics" / f"train_log_{method}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return agent
