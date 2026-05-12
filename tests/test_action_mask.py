from src.rl.action_mask import valid_action_mask

def test_mask(): assert valid_action_mask(0)==[1,0]
