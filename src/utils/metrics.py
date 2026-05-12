from __future__ import annotations

def init_metrics():
    return {k:0.0 for k in ["total_weighted_cost","total_reward","passenger_delay","parcel_lateness","number_late_deliveries","number_undelivered_parcels","terminal_penalty","energy_consumption_kwh","bus_charging_energy_kwh","drone_charging_energy_kwh","power_overload_amount","power_overload_duration","locker_overflow_amount","locker_overflow_duration","minimum_bus_battery_level","battery_safety_violation_frequency","charger_utilization","drone_battery_stockout_frequency","average_locker_holding_time","infeasible_actions","repaired_actions","infeasible_action_rate","action_repair_rate","early_termination","selected_dur","executed_dur","average_selected_charging_duration","average_executed_charging_duration","charging_opportunity_utilization","steps"]}

def finalize_metrics(m: dict):
    steps=max(1,int(m.get('steps',0)))
    m['infeasible_action_rate']=m.get('infeasible_actions',0.0)/steps
    m['action_repair_rate']=m.get('repaired_actions',0.0)/steps
    m['average_selected_charging_duration']=m.get('selected_dur',0.0)/steps
    m['average_executed_charging_duration']=m.get('executed_dur',0.0)/steps
    m['charging_opportunity_utilization']=m['executed_dur']/max(1e-9,m['selected_dur']) if m.get('selected_dur',0)>0 else 0.0
    return m
