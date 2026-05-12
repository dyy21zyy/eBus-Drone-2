from src.low_level.station_operator import operate_station_step


def _station(idle=2, full=2):
    return {
        "drones": [{"id": "d1", "status": "idle"}] * 0 + ([{"id": "d1", "status": "idle"}] if idle >= 1 else []) + ([{"id": "d2", "status": "idle"}] if idle >= 2 else []),
        "locker_parcels": [
            {"id": "p1", "feasible": True, "deadline": 8, "T_out": 3, "T_rt": 5, "c_D": 1, "urgency": 1},
            {"id": "p2", "feasible": True, "deadline": 9, "T_out": 3, "T_rt": 5, "c_D": 2, "urgency": 0},
        ],
        "full_batteries": full,
        "empty_batteries": 0,
        "G_max": 1,
        "P_capacity": 100,
        "P_bat": 10,
    }


def test_cannot_dispatch_without_idle_drone():
    s = _station(idle=0, full=2)
    r = operate_station_step(s, 0, p_e=10, p_l=10, new_parcels=True)
    assert r["n_disp"] == 0


def test_cannot_dispatch_without_full_battery():
    s = _station(idle=2, full=0)
    r = operate_station_step(s, 0, p_e=10, p_l=10, new_parcels=True)
    assert r["n_disp"] == 0


def test_dispatch_updates_locker_and_battery_and_count():
    s = _station(idle=2, full=2)
    r = operate_station_step(s, 0, p_e=10, p_l=10, new_parcels=True)
    assert r["n_disp"] == 2
    assert len(s["locker_parcels"]) == 0
    assert s["full_batteries"] == 0
