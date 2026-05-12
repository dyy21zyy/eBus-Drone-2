class ReplayBuffer:
    def __init__(self): self.data=[]
    def add(self,item): self.data.append(item)
    def __len__(self): return len(self.data)
