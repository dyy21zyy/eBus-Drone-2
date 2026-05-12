from __future__ import annotations

import numpy as np


def evaluate_policy(env, policy, episodes: int = 1, max_steps: int = 100):
    metrics = {"total_weighted_cost": 0.0, "infeasible_actions": 0, "repaired_actions": 0, "steps": 0, "selected_dur": [], "executed_dur": []}
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        for _ in range(max_steps):
            mask = env.get_action_mask()
            action = policy.select_action(obs, mask, {"E_current": env.state.get("battery", 0), "E_min": 0, "E_max": env.state.get("battery_max", 1)})
            if mask[action] == 0:
                metrics["infeasible_actions"] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            metrics["total_weighted_cost"] += -reward
            metrics["steps"] += 1
            metrics["repaired_actions"] += int(info.get("action_repaired", False))
            metrics["selected_dur"].append(info.get("selected_duration", 0))
            metrics["executed_dur"].append(info.get("executed_duration", 0))
            if terminated or truncated:
                break
    steps = max(metrics["steps"], 1)
    metrics["infeasible_action_rate"] = metrics["infeasible_actions"] / steps
    metrics["action_repair_rate"] = metrics["repaired_actions"] / steps
    metrics["average_selected_charging_duration"] = float(np.mean(metrics["selected_dur"])) if metrics["selected_dur"] else 0.0
    metrics["average_executed_charging_duration"] = float(np.mean(metrics["executed_dur"])) if metrics["executed_dur"] else 0.0
    return metrics
