import random
from typing import Optional


def set_seed(seed: int) -> random.Random:
    random.seed(seed)
    return random.Random(seed)


def spawn_rng(seed: int, offset: int = 0) -> random.Random:
    return random.Random(seed + offset * 10007)
