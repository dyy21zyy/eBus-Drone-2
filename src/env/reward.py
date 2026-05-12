from __future__ import annotations


def compute_reward(components: dict, alphas: dict) -> tuple[float, dict]:
    d_p = float(components.get("D_P", 0.0))
    d_l = float(components.get("D_L", 0.0))
    d_e = float(components.get("D_E", 0.0))
    d_pwr = float(components.get("D_Pwr", 0.0))
    d_b = float(components.get("D_B", 0.0))
    d_k = float(components.get("D_K", 0.0))
    total = (
        float(alphas.get("alpha_1", 1.0)) * d_p
        + float(alphas.get("alpha_2", 1.0)) * d_l
        + float(alphas.get("alpha_3", 1.0)) * d_e
        + float(alphas.get("alpha_4", 1.0)) * d_pwr
        + float(alphas.get("alpha_5", 1.0)) * d_b
        + float(alphas.get("alpha_6", 1.0)) * d_k
    )
    rc = {"D_P": d_p, "D_L": d_l, "D_E": d_e, "D_Pwr": d_pwr, "D_B": d_b, "D_K": d_k}
    return -total, rc
