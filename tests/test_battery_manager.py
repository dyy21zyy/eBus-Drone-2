from src.low_level.battery_manager import update_battery

def test_battery(): assert update_battery(0.9,0.2)==1.0
