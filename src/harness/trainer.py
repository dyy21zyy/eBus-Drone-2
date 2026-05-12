from __future__ import annotations

from src.rl.trainer import AgentConfig, DQNAgent


def train_agent(env, method: str = "proposed", episodes: int = 5, max_steps: int = 100):
    obs, _ = env.reset(seed=0)
    agent = DQNAgent(len(obs), len(env.get_action_mask()), AgentConfig(method=method))
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        for _ in range(max_steps):
            mask = env.get_action_mask()
            action = agent.select_action(obs, mask)
            next_obs, reward, term, trunc, info = env.step(action)
            next_mask = env.get_action_mask()
            agent.buffer.add(obs, action, reward, next_obs, term or trunc, mask, next_mask, info)
            agent.update()
            obs = next_obs
            if term or trunc:
                break
    return agent
