from __future__ import annotations

import math
import numpy as np


def _norm(x: float, denom: float) -> float:
    d = max(float(denom), 1e-9)
    return float(x) / d


def _one_hot(ids: list[int], active_id: int) -> list[float]:
    return [1.0 if int(i) == int(active_id) else 0.0 for i in ids]


def build_feature_names(schema: dict) -> list[str]:
    trip_ids = list(schema.get("trip_ids", []))
    station_ids = list(schema.get("station_ids", []))
    stop_ids = list(schema.get("stop_ids", []))
    names = []
    names += [f"g_trip_progress_{tid}" for tid in trip_ids]
    names += [f"g_trip_battery_{tid}" for tid in trip_ids]
    names += [f"g_trip_onboard_pax_{tid}" for tid in trip_ids]
    names += [f"g_trip_onboard_parcels_{tid}" for tid in trip_ids]
    names += [f"g_stop_queue_{sid}" for sid in stop_ids]
    names += [f"g_station_locker_{sid}" for sid in station_ids]
    names += [f"g_station_idle_drones_{sid}" for sid in station_ids]
    names += [f"g_station_full_batteries_{sid}" for sid in station_ids]
    names += [f"g_station_depleted_batteries_{sid}" for sid in station_ids]
    names += [f"g_station_charging_batteries_{sid}" for sid in station_ids]
    names += [f"g_station_power_consumption_{sid}" for sid in station_ids]
    names += [f"g_station_power_margin_{sid}" for sid in station_ids]
    names += [f"g_station_available_chargers_{sid}" for sid in station_ids]
    names += [f"g_station_earliest_charger_release_{sid}" for sid in station_ids]
    names += [f"g_station_earliest_drone_return_{sid}" for sid in station_ids]
    names += [f"g_station_earliest_battery_completion_{sid}" for sid in station_ids]
    names += [f"g_station_urg_min_slack_{sid}" for sid in station_ids]
    names += [f"g_station_urg_avg_slack_{sid}" for sid in station_ids]
    names += [f"g_station_urg_late_count_{sid}" for sid in station_ids]
    names += [f"g_station_urg_risk_count_{sid}" for sid in station_ids]
    names += ["g_future_arrivals_5m", "g_future_arrivals_10m", "g_future_arrivals_15m"]
    names += ["l_time_sin", "l_time_cos"]
    names += [f"l_station_onehot_{sid}" for sid in station_ids]
    names += [f"l_trip_onehot_{tid}" for tid in trip_ids]
    names += ["l_arriving_battery", "l_onboard_before_alight", "l_onboard_parcels_before_unload", "l_alighting", "l_initial_board", "l_unloading_volume", "l_available_chargers", "l_locker", "l_idle_drones", "l_full_batteries", "l_power_margin", "l_urg_min_slack", "l_urg_avg_slack", "l_urg_late_count", "l_urg_risk_count"]
    return names


def build_observation(state: dict, local: dict, schema: dict) -> np.ndarray:
    t = float(state.get("time", 0.0))
    day = max(float(state.get("horizon", 1440.0)), 1.0)
    cyc_sin = math.sin(2 * math.pi * (t % day) / day)
    cyc_cos = math.cos(2 * math.pi * (t % day) / day)

    trip_ids = list(schema.get("trip_ids", []))
    station_ids = list(schema.get("station_ids", []))
    stop_ids = list(schema.get("stop_ids", []))
    batt_cap = float(schema.get("battery_capacity_kwh", 1.0))
    pax_cap = float(schema.get("passenger_capacity", 1.0))
    parcel_cap = float(schema.get("freight_capacity_kg", 1.0))
    q_den = float(schema.get("queue_norm", 50.0))
    horizon = float(schema.get("horizon", day))

    g = []
    g += [_norm(state.get("trip_progress", {}).get(tid, 0.0), max(len(stop_ids) - 1, 1)) for tid in trip_ids]
    g += [_norm(state.get("trip_battery", {}).get(tid, 0.0), batt_cap) for tid in trip_ids]
    g += [_norm(state.get("trip_onboard_pax", {}).get(tid, 0.0), pax_cap) for tid in trip_ids]
    g += [_norm(state.get("trip_onboard_parcels", {}).get(tid, 0.0), parcel_cap) for tid in trip_ids]
    g += [_norm(state.get("stop_queues", {}).get(sid, 0.0), q_den) for sid in stop_ids]
    for key, den in [("station_locker", schema.get("locker_capacity_kg", 1.0)), ("station_idle_drones", schema.get("drones_per_station", 1.0)), ("station_full_batteries", schema.get("battery_inv_norm", 1.0)), ("station_depleted_batteries", schema.get("battery_inv_norm", 1.0)), ("station_charging_batteries", schema.get("battery_inv_norm", 1.0)), ("station_power_consumption", schema.get("station_power_capacity_kw", 1.0)), ("station_power_margin", schema.get("station_power_capacity_kw", 1.0)), ("station_available_chargers", schema.get("chargers_per_station", 1.0)), ("station_earliest_charger_release", horizon), ("station_earliest_drone_return", horizon), ("station_earliest_battery_completion", horizon), ("station_urg_min_slack", horizon), ("station_urg_avg_slack", horizon), ("station_urg_late_count", schema.get("urgency_count_norm", 1.0)), ("station_urg_risk_count", schema.get("urgency_count_norm", 1.0))]:
        g += [_norm(state.get(key, {}).get(stid, 0.0), float(den)) for stid in station_ids]
    g += [_norm(state.get("future_arrivals_5m", 0.0), max(len(trip_ids), 1)), _norm(state.get("future_arrivals_10m", 0.0), max(len(trip_ids), 1)), _norm(state.get("future_arrivals_15m", 0.0), max(len(trip_ids), 1))]

    l = [cyc_sin, cyc_cos]
    l += _one_hot(station_ids, int(local.get("station_id", -1)))
    l += _one_hot(trip_ids, int(local.get("trip_id", -1)))
    l += [_norm(local.get("arriving_battery", 0.0), batt_cap), _norm(local.get("onboard_before_alight", 0.0), pax_cap), _norm(local.get("onboard_parcels_before_unload", 0.0), parcel_cap), _norm(local.get("alighting", 0.0), pax_cap), _norm(local.get("initial_board", 0.0), pax_cap), _norm(local.get("q_f", 0.0), max(float(schema.get("locker_capacity_kg", 1.0)), 1.0)), _norm(local.get("available_chargers", 0.0), float(schema.get("chargers_per_station", 1.0))), _norm(local.get("locker", 0.0), float(schema.get("locker_capacity_kg", 1.0))), _norm(local.get("idle_drones", 0.0), float(schema.get("drones_per_station", 1.0))), _norm(local.get("full_batteries", 0.0), float(schema.get("battery_inv_norm", 1.0))), _norm(local.get("power_margin", 0.0), float(schema.get("station_power_capacity_kw", 1.0))), _norm(local.get("urg_min_slack", 0.0), horizon), _norm(local.get("urg_avg_slack", 0.0), horizon), _norm(local.get("urg_late_count", 0.0), float(schema.get("urgency_count_norm", 1.0))), _norm(local.get("urg_risk_count", 0.0), float(schema.get("urgency_count_norm", 1.0)))]
    return np.asarray(g + l, dtype=np.float32)
