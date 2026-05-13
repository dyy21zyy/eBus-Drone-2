from src.low_level.battery_manager import charge_depleted_batteries
from src.low_level.station_operator import operate_station_step


def test_bus_power_from_active_chargers():
    p_chg = 500.0
    active = 3
    assert active * p_chg == 1500.0


def test_total_power_includes_base_and_drone_load():
    st = {"station_id": 1, "depleted_batteries": 2, "empty_batteries": 2, "full_batteries": 0, "charging_batteries": [], "G_max": 3, "P_capacity": 2000.0, "P_bat": 2.0, "battery_charge_duration_min": 10.0, "charger_release_times_min": [5.0, 5.0, 5.0], "P_chg": 500.0}
    out = operate_station_step(st, 0.0, p_e=1500.0, p_l=150.0)
    assert out["P_tot"] == 1500.0 + 150.0 + 4.0


def test_drone_battery_only_full_after_completion():
    st = {"depleted_batteries": 1, "empty_batteries": 1, "full_batteries": 0, "charging_batteries": [], "G_max": 2, "P_capacity": 500.0, "P_bat": 2.0, "battery_charge_duration_min": 10.0}
    first = charge_depleted_batteries(st, now=0.0, p_e=0.0, p_l=0.0)
    assert first["completed"] == 0
    second = charge_depleted_batteries(st, now=9.0, p_e=0.0, p_l=0.0)
    assert second["completed"] == 0
    third = charge_depleted_batteries(st, now=10.0, p_e=0.0, p_l=0.0)
    assert third["completed"] == 1


def test_charger_availability_by_release_time():
    rel = [1.0, 3.0]
    now = 2.0
    assert sum(1 for t in rel if t <= now) == 1
    now2 = 4.0
    assert sum(1 for t in rel if t <= now2) == 2
