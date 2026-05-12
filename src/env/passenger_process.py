def step_passengers(current:int, arrivals:int, served:int)->int:
    return max(current + arrivals - served, 0)
