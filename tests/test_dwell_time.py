from src.env.dwell_time import compute_dwell_breakdown


def test_dwell_max_logic_and_delay_nonnegative():
    b = compute_dwell_breakdown(2, 3, 1, 2, 4, 30, 1, 1, 1, 10, True)
    assert b.t_s >= b.t_p
    assert b.t_s == b.t_s_hat + 2
    assert b.passenger_delay >= 0


def test_parcel_only_stop_delta_equals_service_time():
    b = compute_dwell_breakdown(0, 0, 1, 2, 5, 0, 1, 1, 1, 0, False)
    assert b.t_f > 0
    assert b.delta_s == b.t_s
