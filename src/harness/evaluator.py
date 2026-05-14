from __future__ import annotations

import csv
from pathlib import Path
from src.utils.metrics import REQUIRED_PAPER_METRICS, init_metrics, finalize_metrics

def _required_component(rc: dict, key: str) -> float:
    if key not in rc:
        raise KeyError(f"Missing reward component in info['reward_components']: {key}")
    return float(rc[key])

def evaluate_policy(env, policy, episodes: int = 1, max_steps: int | None = None):
    metrics = init_metrics()
    min_bat = float("inf")
    termination_reason = None
    terminated_by_env = False
    truncated_by_max_steps = False
    terminal_undelivered_count_total = 0.0
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        step_idx = 0
        while True:
            if max_steps is not None and step_idx >= int(max_steps):
                truncated_by_max_steps = True
                termination_reason = 'max_steps_truncated'
                break
            mask = env.get_action_mask()
            event = getattr(env, 'current_event', None)
            getv = (lambda k, d=0.0: event.get(k, d)) if isinstance(event, dict) else (lambda k, d=0.0: getattr(event, k, d) if event is not None else d)
            info_ctx = {
                'E_current': env.state.get('battery', 0),
                'E_min': env.state.get('battery_min', 0),
                'E_max': env.state.get('battery_max', 1),
                'T_P_est': getv('T_P_est', getv('passenger_dwell_min', 0.0)),
                'T_F': getv('T_F', getv('freight_dwell_min', 0.0)),
            }
            action = policy.select_action(obs, mask, info_ctx)
            step_idx += 1
            if mask[action] == 0:
                metrics['infeasible_actions'] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            rc = info.get('reward_components', {})
            if not rc:
                metrics['total_reward'] += reward
                if terminated or truncated:
                    terminated_by_env = bool(terminated)
                    termination_reason = info.get('termination_reason') or ('truncated' if truncated else termination_reason)
                    break
                continue
            metrics['total_reward'] += reward
            metrics['total_cost'] += _required_component(rc, 'total_cost')
            metrics['onboard_passenger_delay'] += _required_component(rc, 'passenger_delay')
            metrics['parcel_lateness'] += _required_component(rc, 'parcel_lateness')
            metrics['terminal_undelivered_penalty'] += _required_component(rc, 'terminal_penalty')
            metrics['total_energy_consumption'] += _required_component(rc, 'total_energy_kwh')
            metrics['station_power_overload_amount'] += _required_component(rc, 'power_overload')
            metrics['station_power_overload_duration'] += _required_component(rc, 'power_overload_duration')
            metrics['locker_overflow_amount'] += _required_component(rc, 'locker_overflow_amount')
            metrics['locker_overflow_duration'] += _required_component(rc, 'locker_overflow_duration')
            metrics['late_delivery_count'] += float(info.get('late_delivery_count_delta', _required_component(rc, 'number_late_deliveries')))
            metrics['battery_safety_violation_count'] += int(_required_component(rc, 'battery_safety') > 0)
            dwell = info.get('dwell_components', {})
            extra_dwell = float(
                dwell.get(
                    'additional_dwell_min',
                    max(0.0, float(dwell.get('realized_dwell_min', 0.0)) - float(dwell.get('passenger_dwell_min', 0.0))),
                )
            )
            metrics['average_excess_dwell_time'] += extra_dwell
            metrics['total_bus_operating_delay'] += float(info.get('bus_operating_delay_delta', extra_dwell))
            metrics['steps'] += 1
            metrics['repaired_actions'] += int(info.get('action_repaired', False))
            min_bat = min(min_bat, float(env.state.get('battery', 0)))
            if terminated or truncated:
                terminated_by_env = bool(terminated)
                terminal_undelivered_count_total += float(info.get('undelivered_terminal_count', 0.0))
                termination_reason = info.get('termination_reason') or ('truncated' if truncated else termination_reason)
                break

    metrics['minimum_bus_battery'] = 0.0 if min_bat == float('inf') else min_bat
    metrics['undelivered_parcel_count'] = float(terminal_undelivered_count_total) if terminal_undelivered_count_total > 0.0 else (float(sum(1 for p in env.parcel_states.values() if p.get('status') != 'delivered')) if hasattr(env, 'parcel_states') else 0.0)
    end_time = float(getattr(env, 'state', {}).get('time', 0.0))
    delivered_parcels = [
        p for p in getattr(env, 'parcel_states', {}).values()
        if p.get('status') == 'delivered'
        and p.get('delivery_completion_time_min') is not None
        and float(p.get('delivery_completion_time_min')) <= end_time + 1e-9
    ]
    locker_holding_times = [float(p['locker_holding_time_min']) for p in delivered_parcels if p.get('locker_holding_time_min') is not None]
    metrics['average_locker_holding_time'] = float(sum(locker_holding_times)) / float(len(locker_holding_times)) if locker_holding_times else 0.0
    if metrics['total_bus_operating_delay'] <= 0.0:
        metrics['total_bus_operating_delay'] = float(sum(b.get('accumulated_operating_delay_min', 0.0) for b in getattr(env, 'bus_states', {}).values()))
    stations = list(getattr(env, 'station_states', {}).values())
    occupied_time = sum(float(st.get('charger_occupied_time_min', 0.0)) for st in stations)
    charger_capacity_time = sum(float(max(0, int(st.get('charging_slots', len(st.get('charger_release_times_min', [])) or 0)))) * float(getattr(env, 'horizon', getattr(env, 'state', {}).get('horizon', 0.0))) for st in stations)
    metrics['charger_utilization'] = (occupied_time / charger_capacity_time) if charger_capacity_time > 0.0 else 0.0
    # Definition: stockout frequency counts dispatch-trigger opportunities where waiting feasible parcels and idle drones exist but no full battery is available.
    metrics['drone_battery_stockout_count'] = float(sum(float(st.get('dispatch_stockout_count', 0.0)) for st in stations))

    out = finalize_metrics(metrics)
    out['max_steps'] = max_steps
    out['episode_end_time'] = end_time
    out['operating_horizon'] = float(getattr(env, 'horizon', getattr(env, 'state', {}).get('horizon', 0.0)))
    out['termination_reason'] = termination_reason or ('horizon_reached' if out['episode_end_time'] >= out['operating_horizon'] else 'unknown')
    out['full_horizon_completed'] = bool(terminated_by_env and out['termination_reason'] == 'horizon_reached' and not truncated_by_max_steps)
    if out['episode_end_time'] > out['operating_horizon'] + 1e-6:
        raise RuntimeError(f"Episode exceeded operating horizon: {out['episode_end_time']} > {out['operating_horizon']}")
    if out['termination_reason'] == 'horizon_reached' and abs(out['episode_end_time'] - out['operating_horizon']) > 1e-6:
        raise RuntimeError(f"Horizon termination must end at horizon: {out['episode_end_time']} vs {out['operating_horizon']}")
    missing = [k for k in REQUIRED_PAPER_METRICS if k not in out]
    if missing:
        raise KeyError(f"Missing required paper metrics in evaluator output: {missing}")
    return out

def save_eval_metrics(rows: list[dict], out_csv: str):
    p = Path(out_csv); p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('Evaluation produced no rows.')
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
