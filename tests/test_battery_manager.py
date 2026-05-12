from src.low_level.station_operator import operate_station_step


def test_depleted_battery_increases_after_return():
    station = {
        "station_id": 1,
        "drones": [{"drone_id": "d1", "status": "active", "assigned_parcel_id": 10, "return_time": 5}],
        "locker_parcels": [],
        "full_batteries": 1,
        "depleted_batteries": 0,
        "empty_batteries": 0,
        "G_max": 1,
        "charging_batteries": [],
        "battery_charge_duration_min": 2,
        "P_capacity": 100,
        "P_bat": 10,
    }
    operate_station_step(station, 5, parcel_states={}, p_e=95, p_l=10, new_parcels=False)
    assert station["empty_batteries"] == 1
    operate_station_step(station, 6, parcel_states={}, p_e=10, p_l=10, new_parcels=False)
    assert len(station["charging_batteries"]) == 1
    operate_station_step(station, 8, parcel_states={}, p_e=10, p_l=10, new_parcels=False)
    assert station["full_batteries"] >= 2
