class EventCalendar:
    def __init__(self)->None:
        self.events=[]
    def add(self,t:int,name:str)->None:
        self.events.append((t,name)); self.events.sort()
    def pop_next(self):
        return self.events.pop(0) if self.events else None
