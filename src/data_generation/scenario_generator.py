from __future__ import annotations
import random

def generate_scenario(num_stations:int, seed:int)->dict:
    rng=random.Random(seed)
    demand=[rng.randint(0,4) for _ in range(num_stations)]
    return {"num_stations":num_stations,"demand":demand,"seed":seed}
