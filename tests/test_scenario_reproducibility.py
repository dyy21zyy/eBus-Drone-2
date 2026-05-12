from src.utils.config import load_yaml
from src.data_generation.scenario_generator import generate_instance, generate_scenario


def test_reproducible_same_seed():
    cfg = load_yaml('configs/default.yaml')
    inst_cfg = load_yaml('configs/instances/small.yaml')
    inst = generate_instance(cfg, inst_cfg, seed=7)
    s1 = generate_scenario(cfg, inst, 7, 0)
    s2 = generate_scenario(cfg, inst, 7, 0)
    assert s1 == s2


def test_different_seed_changes_scenario():
    cfg = load_yaml('configs/default.yaml')
    inst_cfg = load_yaml('configs/instances/small.yaml')
    inst = generate_instance(cfg, inst_cfg, seed=7)
    s1 = generate_scenario(cfg, inst, 7, 0)
    s2 = generate_scenario(cfg, inst, 8, 0)
    assert s1 != s2
