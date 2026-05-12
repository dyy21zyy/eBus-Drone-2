from src.env.termination import apply_terminal_penalty_once


def test_terminal_penalty_applied_once():
    state = {"terminal_penalty_applied": False}
    undelivered = [{"deadline": 5}, {"deadline": 20}]
    p1 = apply_terminal_penalty_once(state, undelivered, t_end=10, eta_l_term=2, eta_u_term=3)
    p2 = apply_terminal_penalty_once(state, undelivered, t_end=10, eta_l_term=2, eta_u_term=3)
    assert p1 > 0
    assert p2 == 0
