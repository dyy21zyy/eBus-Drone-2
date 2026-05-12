def compute_reward(served:int, waiting:int)->float:
    return float(served - 0.1 * waiting)
