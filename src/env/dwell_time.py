from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DwellBreakdown:
    t_p_hat: float
    t_p: float
    t_f: float
    t_s_hat: float
    t_s: float
    delta_s: float
    passenger_delay: float


def compute_dwell_breakdown(
    n_al: int,
    n_bo0: int,
    tau_q: float,
    tau_e: float,
    q_f: int,
    u_r: float,
    rho_al: float,
    rho_bo: float,
    rho_f: float,
    m_affected: int,
    has_passenger_service: bool,
) -> DwellBreakdown:
    t_p_hat = max(rho_al * max(n_al, 0), rho_bo * max(n_bo0, 0)) if has_passenger_service else 0.0
    t_p = t_p_hat + max(tau_q, 0.0)
    t_f = rho_f * max(q_f, 0)
    t_s_hat = max(t_p, t_f, max(u_r, 0.0))
    excess = 1.0 if max(t_f, max(u_r, 0.0)) > t_p else 0.0
    t_s = t_s_hat + excess * max(tau_e, 0.0)
    if has_passenger_service:
        delta_s = max(0.0, t_s - t_p)
    else:
        delta_s = max(0.0, t_s)
    passenger_delay = max(0.0, m_affected * delta_s)
    return DwellBreakdown(t_p_hat, t_p, t_f, t_s_hat, t_s, delta_s, passenger_delay)
