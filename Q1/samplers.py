import numpy as np
import random
import torch



def distribution1(x, batch_size=512): 
     # Distribution defined as (x, U(0,1)). Can be used for question 3 
     while True: 
         yield(np.array([(x, random.uniform(0, 1)) for _ in range(batch_size)])) 
 
 
 
def distribution2(batch_size): 
     # High dimension uniform distribution 
     while True: 
         yield(np.random.uniform(0, 1, (batch_size, 1))) 
 
 
 
 
def distribution3(batch_size=512): 
     # 1D gaussian distribution 
     while True: 
         yield(np.random.normal(0, 1, (batch_size, 1))) 
 
e = lambda x: np.exp(x) 
tanh = lambda x: (e(x) - e(-x)) / (e(x)+e(-x)) 
def distribution4(batch_size=1): 
     # arbitrary sampler 
     f = lambda x: tanh(x*2+1) + x*0.75 
     while True: 
         yield(f(np.random.normal(0, 1, (batch_size, 1)))) 
 
 
if __name__ == '__main__': 
     
     # Example of usage 
     dist = iter(distribution2(100)) 
   
 