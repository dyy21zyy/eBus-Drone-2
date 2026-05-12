from src.offline.assignment_solver import solve_assignment

def test_offline(): assert solve_assignment([(0,1)])["cost"]==1.0
