class Event: 
    def __init__(self, time, type):
        self.type = type
        self.time = time
    
    def get_type(self):
        return self.type
    
    def get_time(self): 
        return self.time
    
    def __lt__(self, other:object) -> bool:
        return self.time < other.time
    
    
class Queue(Event):
    def __init__(self):
        self.queue = []
  
    def isEmpty(self):
        return len(self.queue) == 0
  
    def enqueue(self, evnt):
        self.queue.append(evnt)
  
    def dequeue(self):
        return self.queue.pop(0)
            
 
        
class PriorityQueue(Event):
    def __init__(self):
        self.queue = []
  
    def isEmpty(self):
        return len(self.queue) == 0
  
    def enqueue(self, evnt):
        self.queue.append(evnt)
        self.queue.sort(reverse=False)
  
    def dequeue(self):
        return self.queue.pop(0)
            
    def get_min(self):
        return self.queue[0].time
    
    