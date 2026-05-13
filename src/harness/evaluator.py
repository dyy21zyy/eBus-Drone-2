from __future__ import annotations

import csv
from pathlib import Path
from src.utils.metrics import REQUIRED_PAPER_METRICS, init_metrics, finalize_metrics


def _required_component(rc: dict, key: str) -> float:
    if key not in rc:
        raise KeyError(f"Missing reward component in info['reward_components']: {key}")
    return float(rc[key])


def evaluate_policy(env, policy, episodes: int = 1, max_steps: int = 100):
    metrics = init_metrics()
    min_bat = float("inf")
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        for _ in range(max_steps):
            mask = env.get_action_mask()
            info_ctx = {"E_current": env.state.get("battery", 0), "E_min": 0, "E_max": env.state.get("battery_max", 1), "T_P_est": 10, "T_F": 10}
            action = policy.select_action(obs, mask, info_ctx)
            if mask[action] == 0:
                metrics["infeasible_actions"] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            rc = info.get("reward_components", {})
            if not rc:
                metrics["total_reward"] += reward
                if terminated or truncated:
                    break
                continue

            metrics["total_reward"] += reward
            metrics["total_cost"] += _required_component(rc, "total_cost")
            metrics["onboard_passenger_delay"] += _required_component(rc, "passenger_delay")
            metrics["parcel_lateness"] += _required_component(rc, "parcel_lateness")
            metrics["terminal_undelivered_penalty"] += _required_component(rc, "terminal_penalty")
            metrics["total_energy_consumption"] += _required_component(rc, "total_energy_kwh")
            metrics["station_power_overload_amount"] += _required_component(rc, "power_overload")
            metrics["station_power_overload_duration"] += _required_component(rc, "power_overload_duration")
            metrics["locker_overflow_amount"] += _required_component(rc, "locker_overflow_amount")
            metrics["locker_overflow_duration"] += _required_component(rc, "locker_overflow_duration")
            metrics["late_delivery_count"] += _required_component(rc, "number_late_deliveries")
            metrics["battery_safety_violation_count"] += int(_required_component(rc, "battery_safety") > 0)

            dwell = info.get("dwell_components", {})
            metrics["average_excess_dwell_time"] += max(0.0, float(dwell.get("realized_dwell_min", 0.0)) - float(dwell.get("passenger_dwell_min", 0.0)))
            metrics["charger_utilization"] += float(info.get("executed_duration_min", 0.0))
            metrics["steps"] += 1
            metrics["repaired_actions"] += int(info.get("action_repaired", False))
            min_bat = min(min_bat, float(env.state.get("battery", 0)))
            if terminated or truncated:
                break

    metrics["minimum_bus_battery"] = 0.0 if min_bat == float("inf") else min_bat
    metrics["undelivered_parcel_count"] = float(sum(1 for p in env.parcel_states.values() if p.get("status") != "delivered")) if hasattr(env, "parcel_states") else 0.0
    metrics["average_locker_holding_time"] = float(sum((p.get("locker_holding_time_min") or 0.0) for p in getattr(env, "parcel_states", {}).values())) / max(1.0, float(len(getattr(env, "parcel_states", {}))))
    metrics["total_bus_operating_delay"] = float(sum(b.get("accumulated_delay_min", 0.0) for b in getattr(env, "bus_states", {}).values()))
    metrics["drone_battery_stockout_count"] = float(sum(1 for st in getattr(env, "station_states", {}).values() if st.get("full_batteries", 0) <= 0))

    out = finalize_metrics(metrics)
    missing = [k for k in REQUIRED_PAPER_METRICS if k not in out]
    if missing:
        raise KeyError(f"Missing required paper metrics in evaluator output: {missing}")
    return out


def save_eval_metrics(rows: list[dict], out_csv: str):
    p = Path(out_csv); p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Evaluation produced no rows.")
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
