from __future__ import annotations


REQUIRED_PAPER_METRICS = [
    "total_cost","total_reward","onboard_passenger_delay","average_excess_dwell_time",
    "total_bus_operating_delay","parcel_lateness","late_delivery_count","undelivered_parcel_count",
    "average_locker_holding_time","terminal_undelivered_penalty","minimum_bus_battery",
    "battery_safety_violation_count","total_energy_consumption","station_power_overload_amount",
    "station_power_overload_duration","locker_overflow_amount","locker_overflow_duration",
    "charger_utilization","drone_battery_stockout_count",
]

REQUIRED_VALIDATION_FIELDS = [
    "episode_end_time",
    "operating_horizon_min",
    "termination_reason",
    "full_horizon_completed",
    "truncated_by_max_steps",
]


def init_metrics():
    m = {k: 0.0 for k in REQUIRED_PAPER_METRICS}
    m.update({"steps": 0.0, "infeasible_actions": 0.0, "repaired_actions": 0.0, "requested_action_sum": 0.0, "executed_action_sum": 0.0, "action_gap_sum": 0.0, "invalid_action_count": 0.0, "action_repair_count": 0.0})
    return m


def finalize_metrics(m: dict):
    steps = max(1, int(m.get("steps", 0)))
    m["average_excess_dwell_time"] = float(m.get("average_excess_dwell_time", 0.0)) / steps
    m["infeasible_action_rate"] = m.get("infeasible_actions", 0.0) / steps
    repairs = float(m.get("action_repair_count", m.get("repaired_actions", 0.0)))
    m["action_repair_rate"] = repairs / steps
    m["mean_requested_action"] = float(m.get("requested_action_sum", 0.0)) / steps
    m["mean_executed_action"] = float(m.get("executed_action_sum", 0.0)) / steps
    m["mean_action_gap_after_repair"] = float(m.get("action_gap_sum", 0.0)) / steps
    return m
