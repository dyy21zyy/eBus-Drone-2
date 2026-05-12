def update_battery(level:float, delta:float)->float:
    return min(1.0, max(0.0, level + delta))
