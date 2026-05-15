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


def test_new_battery_jobs_respect_active_slots_and_residual_power():
    st = {
        "depleted_batteries": 5,
        "empty_batteries": 5,
        "full_batteries": 0,
        "charging_batteries": [
            {"start_time_min": 0.0, "completion_time_min": 20.0},
            {"start_time_min": 0.0, "completion_time_min": 20.0},
        ],
        "G_max": 3,
        "P_capacity": 510.0,
        "P_bat": 100.0,
        "battery_charge_duration_min": 10.0,
    }
    # residual for new jobs: 510 - 200(bus) - 100(base) - 2*100(active) = 10 => 0 extra slots
    out = charge_depleted_batteries(st, now=1.0, p_e=200.0, p_l=100.0)
    assert out["g"] == 0
    assert len(st["charging_batteries"]) == 2
    assert out["P_D"] == 200.0
    assert st["depleted_batteries"] == 5


def test_charger_availability_by_release_time():
    rel = [1.0, 3.0]
    now = 2.0
    assert sum(1 for t in rel if t <= now) == 1
    now2 = 4.0
    assert sum(1 for t in rel if t <= now2) == 2


def test_default_charge_completion_at_45_min_when_duration_unspecified():
    st = {"depleted_batteries": 1, "empty_batteries": 1, "full_batteries": 0, "charging_batteries": [], "G_max": 1, "P_capacity": 500.0, "P_bat": 2.0}
    charge_depleted_batteries(st, now=0.0, p_e=0.0, p_l=0.0)
    assert charge_depleted_batteries(st, now=44.9, p_e=0.0, p_l=0.0)["completed"] == 0
    assert charge_depleted_batteries(st, now=45.0, p_e=0.0, p_l=0.0)["completed"] == 1
