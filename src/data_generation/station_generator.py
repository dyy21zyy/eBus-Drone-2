
def select_integrated_stations(default_station_ids: list[int], num_integrated_stations: int) -> list[int]:
    return sorted(default_station_ids[:num_integrated_stations])
