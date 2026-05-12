def build_assignment_data(scenario:dict)->list[tuple[int,int]]:
    return list(enumerate(scenario.get("demand",[])))
