from src.low_level.station_power import station_power_balance

def test_power(): assert station_power_balance(3,5)==2
