from src.env.termination import smoke

def test_term(): assert smoke().endswith("termination")
