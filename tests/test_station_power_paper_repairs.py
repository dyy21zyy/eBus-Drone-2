import pytest

from src.env.ebus_drone_env import EBusDroneEnv
from src.low_level.battery_manager import charge_depleted_batteries


def test_15_seconds_bus_charge_energy_exact_kwh():
    env = EBusDroneEnv(smoke_test=True)
    st = env.station_states[1]
    st["bus_charging_intervals"] = [{"start_time_min": 0.0, "end_time_min": 0.25, "charging_power_kw": 500.0}]
    st["charger_release_times_min"] = [0.25]
    env._run_station_interval(0.0, 0.25)
    assert st.get("bus_charging_energy_kwh", 0.0) == pytest.approx(500.0 * 0.25 / 60.0, rel=1e-8)


def test_two_bus_charges_can_overload_softly():
    env = EBusDroneEnv(smoke_test=True)
    st = env.station_states[1]
    st["P_capacity"] = 800.0
    st["base_load_profile_kw"] = [0.0, 0.0]
    st["bus_charging_intervals"] = [
        {"start_time_min": 0.0, "end_time_min": 1.0, "charging_power_kw": 500.0},
        {"start_time_min": 0.0, "end_time_min": 1.0, "charging_power_kw": 500.0},
    ]
    env._run_station_interval(0.0, 1.0)
    assert st.get("power_overload_amount_kw_min", 0.0) > 0.0


def test_ongoing_battery_charging_non_preemptive_when_bus_starts():
    st = {
        "depleted_batteries": 3, "empty_batteries": 3, "full_batteries": 0, "charging_batteries": [],
        "G_max": 2, "P_capacity": 250.0, "P_bat": 100.0, "battery_charge_duration_min": 10.0,
    }
    out0 = charge_depleted_batteries(st, now=0.0, p_e=0.0, p_l=0.0)
    assert out0["g"] == 2
    out1 = charge_depleted_batteries(st, now=1.0, p_e=200.0, p_l=0.0)
    assert len(st["charging_batteries"]) == 2
    assert out1["g"] == 0


def test_base_load_not_counted_in_controllable_energy_cost():
    env = EBusDroneEnv(smoke_test=True)
    st = env.station_states[1]
    st["base_load_profile_kw"] = [1000.0, 1000.0]
    env._run_station_interval(0.0, 1.0)
    snap = env._snapshot_cumulative_metrics()
    assert snap["energy_consumption"] == pytest.approx(
        snap["bus_charging_energy_kwh"] + snap["drone_charging_energy_kwh"], rel=1e-9
    )


def test_base_load_profile_treated_piecewise_constant():
    env = EBusDroneEnv(smoke_test=True)
    st = env.station_states[1]
    st["P_capacity"] = 1e9
    st["base_load_profile_kw"] = [0.0, 60.0]
    st["depleted_batteries"] = 1
    st["empty_batteries"] = 1
    st["G_max"] = 1
    st["P_bat"] = 60.0
    st["charging_batteries"] = []
    env._run_station_interval(0.0, 1.1)
    # charging can start before t=1.0; after base-load jump to 60kW no new jobs can start but active one continues.
    assert len(st.get("charging_batteries", [])) == 1
