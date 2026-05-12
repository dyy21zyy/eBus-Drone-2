from .passenger_process import step_passengers
from .reward import compute_reward

class EBusDroneEnv:
    def __init__(self):
        self.waiting=0
    def reset(self):
        self.waiting=0
        return {"waiting":self.waiting}
    def step(self, action:int):
        served=1 if action>0 and self.waiting>0 else 0
        self.waiting=step_passengers(self.waiting, arrivals=1, served=served)
        r=compute_reward(served, self.waiting)
        return {"waiting":self.waiting}, r, False, {}
