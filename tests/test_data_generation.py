from src.data_generation.scenario_generator import generate_scenario

def test_gen(): assert generate_scenario(3,1)["num_stations"]==3
