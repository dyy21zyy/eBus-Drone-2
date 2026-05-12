from src.env.dwell_time import compute_dwell_breakdown


def test_dwell_max_logic_and_delay_nonnegative():
    b = compute_dwell_breakdown(2, 3, 1, 2, 4, 30, 1, 1, 1, 10, True)
    assert b.t_s >= b.t_p
    assert b.t_s == b.t_s_hat + 2
    assert b.passenger_delay >= 0
