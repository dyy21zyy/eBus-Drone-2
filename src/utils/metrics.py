from __future__ import annotations

def init_metrics():
    return {k:0.0 for k in [
        "total_weighted_cost","total_reward","total_passenger_delay","total_parcel_lateness","number_late_deliveries","number_undelivered_parcels","terminal_penalty",
        "total_energy_kwh","bus_charging_energy_kwh","drone_charging_energy_kwh","power_overload_amount","power_overload_duration",
        "locker_overflow_amount","locker_overflow_duration","minimum_bus_battery_level","battery_safety_violation_count","charger_utilization",
        "drone_battery_stockout_count","average_locker_holding_time","average_executed_charging_duration","number_decision_events",
        "infeasible_actions","repaired_actions","infeasible_action_rate","action_repair_rate","early_termination","selected_dur","executed_dur","steps"
    ]}

def finalize_metrics(m: dict):
    steps=max(1,int(m.get('steps',0)))
    m['infeasible_action_rate']=m.get('infeasible_actions',0.0)/steps
    m['action_repair_rate']=m.get('repaired_actions',0.0)/steps
    m['average_executed_charging_duration']=m.get('executed_dur',0.0)/steps
    return m
