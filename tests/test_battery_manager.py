from src.low_level.station_operator import operate_station_step


def test_depleted_battery_increases_after_return():
    station = {
        "drones": [{"id": "d1", "status": "active", "return_time": 5}],
        "locker_parcels": [],
        "full_batteries": 1,
        "empty_batteries": 0,
        "G_max": 0,
        "P_capacity": 100,
        "P_bat": 10,
    }
    operate_station_step(station, 5, p_e=10, p_l=10, new_parcels=False)
    assert station["empty_batteries"] == 1
