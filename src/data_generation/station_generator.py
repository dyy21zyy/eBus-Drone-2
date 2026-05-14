from __future__ import annotations


def generate_stations(config: dict, instance_cfg: dict) -> dict:
    ids = list(instance_cfg["station_ids"])
    stop_ids_cfg = instance_cfg.get("station_stop_ids", ids)
    if len(stop_ids_cfg) != len(ids):
        raise ValueError("station_stop_ids must have same length as station_ids")
    stations = []
    for sid, stop_id in zip(ids, stop_ids_cfg):
        stations.append({
            "station_id": int(sid),
            "stop_id": int(stop_id),
            "chargers": int(config["charging"]["chargers_per_station"]),
            "pantograph_power_kw": float(config["charging"]["pantograph_power_kw"]),
            "locker_capacity_kg": float(config["parcel"]["locker_capacity_kg"]),
            "station_power_capacity_kw": float(config["power"]["station_capacity_kw"]),
            "drones": int(config["drone"]["drones_per_station"]),
            "initial_fully_charged_batteries": int(config["battery"]["initial_fully_charged_per_station"]),
            "initial_depleted_batteries": int(config["battery"]["initial_depleted_per_station"]),
        })
    return {"station_ids": ids, "stations": stations}
