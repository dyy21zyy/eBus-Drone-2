from src.low_level.station_operator import operate_station_step


def _station(idle=2, full=2, locker_parcels=None, parcels=None):
    if parcels is None:
        parcels = {
            1: {"id": 1, "parcel_id": 1, "status": "in_locker", "assigned_station_id": 1, "deadline_min": 8, "release_time_min": 0.0, "T_out_min": 3, "T_rt_min": 5, "drone_cost": 1, "weight_kg": 1.0},
            2: {"id": 2, "parcel_id": 2, "status": "in_locker", "assigned_station_id": 1, "deadline_min": 9, "release_time_min": 0.0, "T_out_min": 3, "T_rt_min": 5, "drone_cost": 2, "weight_kg": 2.0},
        }
    if locker_parcels is None:
        locker_parcels = list(parcels.keys())
    drones = []
    for i in range(idle):
        drones.append({"drone_id": f"d{i+1}", "status": "idle", "assigned_parcel_id": None, "return_time": None})
    return {
        "station_id": 1,
        "drones": drones,
        "locker_parcels": locker_parcels,
        "full_batteries": full,
        "depleted_batteries": 0,
        "empty_batteries": 0,
        "G_max": 1,
        "charging_batteries": [],
        "P_capacity": 100,
        "P_bat": 10,
        "locker_inventory_kg": float(sum(float(parcels[i]["weight_kg"]) for i in locker_parcels)),
    }, parcels


def test_cannot_dispatch_without_idle_drone():
    s, parcels = _station(idle=0, full=2)
    r = operate_station_step(s, 0, parcel_states=parcels, p_e=10, p_l=10, new_parcels=True)
    assert r["n_disp"] == 0


def test_cannot_dispatch_without_full_battery():
    s, parcels = _station(idle=2, full=0)
    r = operate_station_step(s, 0, parcel_states=parcels, p_e=10, p_l=10, new_parcels=True)
    assert r["n_disp"] == 0


def test_dispatch_count_limited_by_full_batteries_and_unique_assignments():
    parcels = {
        i: {"id": i, "parcel_id": i, "status": "in_locker", "assigned_station_id": 1, "deadline_min": 50 + i, "release_time_min": 0.0, "T_out_min": 3, "T_rt_min": 6, "drone_cost": 1, "weight_kg": 1.0}
        for i in range(1, 6)
    }
    s, parcels = _station(idle=3, full=2, locker_parcels=[1, 2, 3, 4, 5], parcels=parcels)
    r = operate_station_step(s, 0, parcel_states=parcels, p_e=10, p_l=10, new_parcels=True)
    assert r["n_disp"] == 2
    selected_parcels = [a["parcel_id"] for a in r["assignments"]]
    selected_drones = [a["drone_id"] for a in r["assignments"]]
    assert len(selected_parcels) == len(set(selected_parcels))
    assert len(selected_drones) == len(set(selected_drones))


def test_urgent_or_late_priority_when_base_cost_similar():
    parcels = {
        1: {"id": 1, "parcel_id": 1, "status": "in_locker", "assigned_station_id": 1, "delivery_deadline_min": 4, "release_time_min": 0.0, "T_out_min": 5, "T_rt_min": 8, "drone_cost": 1.0, "weight_kg": 1.0},
        2: {"id": 2, "parcel_id": 2, "status": "in_locker", "assigned_station_id": 1, "delivery_deadline_min": 30, "release_time_min": 0.0, "T_out_min": 5, "T_rt_min": 8, "drone_cost": 1.0, "weight_kg": 1.0},
    }
    s, parcels = _station(idle=1, full=1, locker_parcels=[1, 2], parcels=parcels)
    r = operate_station_step(s, 0, parcel_states=parcels, p_e=10, p_l=10, new_parcels=True)
    assert r["n_disp"] == 1
    assert r["assignments"][0]["parcel_id"] == 1


def test_dispatch_updates_locker_and_battery_and_count():
    s, parcels = _station(idle=2, full=2)
    r = operate_station_step(s, 0, parcel_states=parcels, p_e=10, p_l=10, new_parcels=True)
    assert r["n_disp"] == 2
    assert len(s["locker_parcels"]) == 0
    assert s["full_batteries"] == 0
    assert parcels[1]["status"] == "assigned_to_drone"
    assert parcels[1]["pickup_time_min"] == 0
    assert parcels[1]["delivery_completion_time_min"] == 3
    assert parcels[1]["drone_return_time_min"] == 5


def test_drone_return_adds_depleted_battery_and_realized_completion_recorded():
    s, parcels = _station(idle=1, full=1)
    operate_station_step(s, 0, parcel_states=parcels, p_e=10, p_l=10, new_parcels=True)
    assert s["depleted_batteries"] == 0
    operate_station_step(s, 3, parcel_states=parcels, p_e=10, p_l=10, new_parcels=False)
    assert parcels[1]["status"] == "delivered"
    assert parcels[1]["delivery_completion_time_min"] == 3
    operate_station_step(s, 5, parcel_states=parcels, p_e=10, p_l=10, new_parcels=False)
    assert s["depleted_batteries"] + len(s.get("charging_batteries", [])) >= 1


def test_not_dispatch_if_not_released():
    s, parcels = _station(idle=1, full=1)
    parcels[1]["status"] = "onboard"
    r = operate_station_step(s, 0, parcel_states=parcels, p_e=10, p_l=10, new_parcels=True)
    assert r["n_disp"] == 1
    assert parcels[1]["status"] == "onboard"
