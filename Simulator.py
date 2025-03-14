import numpy as np
import pandas as pd

from Events import *

class Simulator: 
    def __init__(self):
        self.clock = 0
        self.number_in_service = 0
        self.queue_length = 0
        self.last_event_time = 0.0
        self.total_busy = 0.0
        self.max_queue_length = 0
        self.sum_response_time = 0.0
        self.number_of_departure = 0.0
        self.sum_wait_time = 0.0
        self.weighted_queue_length = 0.0
        
        self.number_of_customers = 0
        
       # self.busy_servers = 0
        
        self.customers = Queue()
        self.future_event_list = PriorityQueue()
        
        evt = Event(self.clock + self.gen_int_arr(), 'A') 
        self.future_event_list.enqueue(evt)
    
    def process_arrival(self, evt):
        self.customers.enqueue(evt)
        
        self.weighted_queue_length += (self.clock - self.last_event_time) * self.queue_length
        self.queue_length += 1
        
        if self.number_in_service == 0:
            self.schedule_departure()
            
        else:
            self.total_busy += (self.clock - self.last_event_time)
            
        if self.max_queue_length < self.queue_length:
            self.max_queue_length = self.queue_length
            
            
        evt = Event(self.clock + self.gen_int_arr(), 'A') 
        self.future_event_list.enqueue(evt)
        
        self.last_event_time = self.clock
               
    
    def schedule_departure(self):
        self.number_of_customers += 1
        self.sum_wait_time += self.clock - self.customers.queue[0].time
        
        evt = Event(self.clock + self.gen_service_time(),'D')
        self.future_event_list.enqueue(evt)
        
        self.number_in_service = 1
        self.queue_length -= 1
        
        
    def process_departure(self, e):
        
        evt = self.customers.dequeue()
        
        self.weighted_queue_length += (self.clock - self.last_event_time) * self.queue_length
        
        if self.queue_length > 0:
            self.schedule_departure()
        else:
            self.number_in_service = 0
            
        self.sum_response_time += self.clock - evt.get_time()
        
        self.total_busy += self.clock - self.last_event_time
        self.number_of_departure += 1
            
        self.last_event_time = self.clock
    
    def gen_int_arr(self):                                            
        return np.random.exponential(10.0)
            
    def gen_service_time(self):                               
        return np.random.exponential(5.0)

    def report_generation(self):
        print("Clock:", self.clock)
        print("Server utilization is: ", self.total_busy/self.clock)
        print("Average waiting time in queue is: ", self.sum_wait_time / self.number_of_customers)
        print("Average number of customers in quque: ", self.weighted_queue_length / self.clock)
        print("Maximum nymber of customers in queue is: ", self.max_queue_length)
    

def main():
    ss = Simulator()
    
    while(ss.clock < 1000000):
        evt = ss.future_event_list.dequeue()
        ss.clock = evt.get_time()
        
        if (evt.get_type() == 'A'):
            ss.process_arrival(evt)
        else:
            ss.process_departure(evt)
   
    ss.report_generation()

    
if __name__ == '__main__':
    main()