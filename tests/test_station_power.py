from src.low_level.station_power import compute_station_power


def test_station_power_total_and_overload():
    out = compute_station_power(20, 15, 10, 40)
    assert out["P_tot"] == out["P_E"] + out["P_D"] + out["P_L"]
    assert out["overload"] == 5
