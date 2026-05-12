import copy, random, numpy as np, torch
import torch.nn.functional as F
from src.rl.agents.am_ddqn_dr_agent import AMDDQNDRAgent
from src.rl.network_factory import build_network


class AMDuelingDDQNDRAgent(AMDDQNDRAgent):
    def __init__(self, obs_dim, action_dim, cfg):
        super().__init__(obs_dim, action_dim, cfg)
        self.online = build_network(obs_dim, action_dim, dueling=True).to(self.device)
        self.target = copy.deepcopy(self.online)
        self.optim = torch.optim.Adam(self.online.parameters(), lr=float(cfg.get('lr',1e-3)))

    def select_action(self, observation, action_mask, training=True) -> int:
        feasible=[i for i,v in enumerate(action_mask) if v>0]
        if training and random.random()<self._eps(): return random.choice(feasible) if feasible else 0
        with torch.no_grad():
            obs=torch.tensor(np.asarray(observation),dtype=torch.float32,device=self.device).unsqueeze(0)
            m=torch.tensor(np.asarray(action_mask),dtype=torch.float32,device=self.device).unsqueeze(0)
            q=self.online(obs,m)[0].masked_fill(m[0]<=0,-1e9)
        return int(torch.argmax(q).item())

    def update(self,batch_size=None):
        bs=batch_size or int(self.cfg.get('batch_size',32))
        if len(self.buffer)<max(bs,int(self.cfg.get('warmup_steps',20))): return None
        b=self.buffer.sample(bs)
        obs=torch.tensor(np.stack([t.observation for t in b]),dtype=torch.float32,device=self.device)
        act=torch.tensor([t.action_index for t in b],dtype=torch.long,device=self.device)
        rew=torch.tensor([t.reward for t in b],dtype=torch.float32,device=self.device)
        nxt=torch.tensor(np.stack([t.next_observation for t in b]),dtype=torch.float32,device=self.device)
        nmask=torch.tensor(np.stack([t.next_action_mask for t in b]),dtype=torch.float32,device=self.device)
        done=torch.tensor([t.done for t in b],dtype=torch.float32,device=self.device)
        q=self.online(obs,None).gather(1,act.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_on=self.online(nxt,nmask).masked_fill(nmask<=0,-1e9)
            a_next=q_on.argmax(dim=1)
            q_next=self.target(nxt,nmask).gather(1,a_next.unsqueeze(1)).squeeze(1)
            y=rew+(1-done)*float(self.cfg.get('gamma',0.99))*q_next
        loss=F.mse_loss(q,y); self.optim.zero_grad(); loss.backward(); self.optim.step(); self.steps+=1
        if self.steps % int(self.cfg.get('target_update',25))==0: self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())
