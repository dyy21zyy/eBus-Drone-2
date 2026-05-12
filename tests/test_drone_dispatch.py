from src.low_level.station_operator import operate_station_step


def _station(idle=2, full=2):
    parcels = {
        1: {"id": 1, "status": "in_locker", "assigned_station_id": 1, "deadline_min": 8, "T_out": 3, "T_rt": 5, "c_D": 1},
        2: {"id": 2, "status": "in_locker", "assigned_station_id": 1, "deadline_min": 9, "T_out": 3, "T_rt": 5, "c_D": 2},
    }
    return {
        "station_id": 1,
        "drones": ([{"drone_id": "d1", "status": "idle", "assigned_parcel_id": None, "return_time": None}] if idle >= 1 else []) + ([{"drone_id": "d2", "status": "idle", "assigned_parcel_id": None, "return_time": None}] if idle >= 2 else []),
        "locker_parcels": [1, 2],
        "full_batteries": full,
        "depleted_batteries": 0,
        "empty_batteries": 0,
        "G_max": 1,
        "charging_batteries": [],
        "P_capacity": 100,
        "P_bat": 10,
    }, parcels


def test_cannot_dispatch_without_idle_drone():
    s, parcels = _station(idle=0, full=2)
    r = operate_station_step(s, 0, parcel_states=parcels, p_e=10, p_l=10, new_parcels=True)
    assert r["n_disp"] == 0


def test_cannot_dispatch_without_full_battery():
    s, parcels = _station(idle=2, full=0)
    r = operate_station_step(s, 0, parcel_states=parcels, p_e=10, p_l=10, new_parcels=True)
    assert r["n_disp"] == 0


def test_dispatch_updates_locker_and_battery_and_count():
    s, parcels = _station(idle=2, full=2)
    r = operate_station_step(s, 0, parcel_states=parcels, p_e=10, p_l=10, new_parcels=True)
    assert r["n_disp"] == 2
    assert len(s["locker_parcels"]) == 0
    assert s["full_batteries"] == 0
    assert parcels[1]["status"] == "assigned_to_drone"
