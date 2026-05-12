class BasePolicy:
    def act(self, mask:list[int])->int:
        return next((i for i,v in enumerate(mask) if v), 0)
