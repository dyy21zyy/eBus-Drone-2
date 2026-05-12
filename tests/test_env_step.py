from src.env.ebus_drone_env import EBusDroneEnv

def test_step(): env=EBusDroneEnv(); env.reset(); s,r,d,i=env.step(0); assert "waiting" in s
