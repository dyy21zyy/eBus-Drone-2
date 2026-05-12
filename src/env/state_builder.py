from __future__ import annotations

import math
import numpy as np


def build_observation(state: dict, local: dict) -> np.ndarray:
    t = float(state["time"])
    day = max(float(state.get("horizon", 1440.0)), 1.0)
    cyc_sin = math.sin(2 * math.pi * (t % day) / day)
    cyc_cos = math.cos(2 * math.pi * (t % day) / day)

    global_features = [
        float(state["trip_location"]),
        float(state["battery"]),
        float(state["onboard_passengers"]),
        float(state["onboard_parcels"]),
        float(state["queue"]),
        float(state["locker"]),
        float(state["idle_drones"]),
        float(state["full_batteries"]),
        float(state["station_power"]),
        float(state["power_margin"]),
        float(state["available_chargers"]),
        float(state["parcel_urgency"]),
        float(state["calendar_len"]),
    ]
    local_features = [
        cyc_sin,
        cyc_cos,
        float(local["station_id"]),
        float(local["trip_id"]),
        float(local["arriving_battery"]),
        float(local["onboard_before_alight"]),
        float(local["onboard_parcels_before_unload"]),
        float(local.get("onboard_parcel_weight_before_unload", 0.0)),
        float(local["alighting"]),
        float(local["initial_board"]),
        float(local["q_f"]),
        float(local.get("num_unloading_parcels", 0.0)),
        float(local["available_chargers"]),
        float(local["locker"]),
        float(local["idle_drones"]),
        float(local["full_batteries"]),
        float(local["power_margin"]),
        float(local["delivery_urgency"]),
    ]
    return np.asarray(global_features + local_features, dtype=np.float32)
