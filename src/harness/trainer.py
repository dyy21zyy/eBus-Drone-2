from __future__ import annotations

from pathlib import Path
import json

from src.rl.agents.dqn_dr_agent import DQNDRAgent
from src.rl.agents.ddqn_dr_agent import DDQNDRAgent
from src.rl.agents.am_ddqn_dr_agent import AMDDQNDRAgent
from src.rl.agents.am_dueling_ddqn_dr_agent import AMDuelingDDQNDRAgent


def train_agent(env, method: str = "proposed", episodes: int = 5, max_steps: int = 100, smoke_test: bool = False, out_root: str = "outputs"):
    obs, _ = env.reset(seed=0)
    cfg = {"gamma":0.99,"lr":1e-3,"batch_size":8 if smoke_test else 32,"target_update":10,"warmup_steps":4 if smoke_test else 20,"epsilon_start":1.0,"epsilon_end":0.05,"epsilon_decay":50 if smoke_test else 500,"capacity":2000}
    cls = {"dqn_dr": DQNDRAgent, "ddqn_dr": DDQNDRAgent, "am_ddqn_dr": AMDDQNDRAgent, "proposed": AMDuelingDDQNDRAgent, "am_dueling_ddqn_dr": AMDuelingDDQNDRAgent}.get(method, AMDuelingDDQNDRAgent)
    agent = cls(len(obs), len(env.get_action_mask()), cfg)
    curve = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep); ep_reward=0.0
        for _ in range(max_steps):
            mask=env.get_action_mask(); action=agent.select_action(obs, mask, training=True)
            nxt, reward, term, trunc, info=env.step(action)
            nxt_mask=env.get_action_mask() if not(term or trunc) else mask
            agent.observe(obs, action, reward, nxt, term or trunc, mask, nxt_mask, info)
            agent.update(); obs=nxt; ep_reward += reward
            if term or trunc: break
        curve.append({"episode":ep,"reward":ep_reward})
    out=Path(out_root); (out/"checkpoints").mkdir(parents=True,exist_ok=True); (out/"metrics").mkdir(parents=True,exist_ok=True)
    ckpt=out/"checkpoints"/f"{method}.pt"; agent.save_checkpoint(str(ckpt))
    (out/"metrics"/f"train_curve_{method}.json").write_text(json.dumps(curve,indent=2),encoding='utf-8')
    return agent
