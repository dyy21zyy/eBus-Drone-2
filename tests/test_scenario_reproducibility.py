from src.data_generation.scenario_generator import generate_scenario

def test_repro(): assert generate_scenario(4,7)==generate_scenario(4,7)
