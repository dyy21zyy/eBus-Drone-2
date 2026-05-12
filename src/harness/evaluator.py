from __future__ import annotations

from src.utils.metrics import init_metrics, finalize_metrics


def evaluate_policy(env, policy, episodes: int = 1, max_steps: int = 100):
    metrics = init_metrics()
    min_bat = float('inf')
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        for _ in range(max_steps):
            mask = env.get_action_mask()
            info_ctx={"E_current": env.state.get("battery", 0), "E_min": 0, "E_max": env.state.get("battery_max", 1), "T_P_est":10, "T_F":10}
            action = policy.select_action(obs, mask, info_ctx)
            if mask[action] == 0: metrics["infeasible_actions"] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            rc=info.get('reward_components',{})
            metrics['total_reward'] += reward; metrics['total_weighted_cost'] += rc.get('total_cost',-reward)
            metrics['passenger_delay'] += rc.get('D_P',0); metrics['parcel_lateness'] += rc.get('D_L',0); metrics['terminal_penalty'] += rc.get('terminal_penalty',0)
            metrics['energy_consumption_kwh'] += rc.get('D_E',0); metrics['bus_charging_energy_kwh'] += rc.get('D_E',0); metrics['drone_charging_energy_kwh'] += 0
            metrics['power_overload_amount'] += rc.get('D_Pwr',0); metrics['locker_overflow_amount'] += rc.get('D_K',0)
            metrics['steps'] += 1; metrics['repaired_actions'] += int(info.get('action_repaired', False))
            metrics['selected_dur'] += info.get('selected_duration',0); metrics['executed_dur'] += info.get('executed_duration',0)
            min_bat=min(min_bat,float(env.state.get('battery',0)))
            if terminated or truncated:
                metrics['early_termination'] += int(terminated)
                break
    metrics['minimum_bus_battery_level']=0.0 if min_bat==float('inf') else min_bat
    return finalize_metrics(metrics)
