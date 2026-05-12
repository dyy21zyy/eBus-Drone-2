from src.env.reward import compute_reward


def test_reward_components_returned():
    r, comps = compute_reward({"D_P": 1, "D_L": 2, "D_E": 3, "D_Pwr": 4, "D_B": 5, "D_K": 6},
                              {"alpha_1": 1, "alpha_2": 1, "alpha_3": 1, "alpha_4": 1, "alpha_5": 1, "alpha_6": 1})
    assert r == -(1 + 2 + 3 + 4 + 5 + 6)
    assert set(comps.keys()) == {"D_P", "D_L", "D_E", "D_Pwr", "D_B", "D_K"}
