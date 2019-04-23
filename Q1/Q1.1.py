# -*- coding: utf-8 -*-
"""

Created on Thu Apr 18 17:09:23 2019

@author: binoS
"""


from __future__ import print_function
import numpy as np
import torch 

import matplotlib.pyplot as plt
import torch.nn as nn 
import torch.optim as optim 

from torch.autograd import Variable 
import samplers

cuda = torch.cuda.is_available()
if cuda:
    gpu = 0

class Discriminant(nn.Module):

    def __init__(self,input_shape, hidden_shape):
        super(Discriminant, self).__init__()

        
        self.layer1 = nn.Linear(input_shape, hidden_shape)
        self.relu1 = nn.ReLU()      
        self.layer2 = nn.Linear(hidden_shape, hidden_shape)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_shape, 1)
        self.smd = nn.Sigmoid()
        
    
 
    def forward(self, input):
        output = self.layer1(input)
        output = self.relu1(output)
        output = self.layer2(output)
        output = self.relu2(output)
        output = self.layer3(output)
        output = self.smd(output)
        return output#.view(shape) # (BATCH_SIZE, SEQ_LEN, len(charmap))

def dist(choice,size,phi):
    if choice ==1:
        dist = iter(samplers.distribution1(phi,size)) 
        return dist
    if choice ==3:
        dist = iter(samplers.distribution3(size)) 
        return dist 
    if choice ==4:
        dist = iter(samplers.distribution4(size)) 
        return dist 

    
def JS_estimator(epoch, choice_p,choice_q,BATCHSIZE, hidden_size, d_iter,phi ):

    modelD = Discriminant(2,hidden_size)
    if cuda:
        modelD = modelD.cuda(gpu)
        
    optimizerD = optim.SGD(modelD.parameters(), lr=1e-2) # ,weight_decay=0.001
    #nn.CrossEntropyLoss()
    p = dist(choice_p,BATCHSIZE,0)
    q = dist(choice_q,BATCHSIZE,phi)
    for iter in range(epoch):
        
        for d_it in range(d_iter):
            modelD.train()
            optimizerD.zero_grad()  
           
            tensorp = torch.Tensor(next(p))
            if cuda:
                tensorp = tensorp.cuda(gpu)
            
            #tensorp = Variable(tensorp, requires_grad=True) #torch.randn(BATCH_SIZE,1)*0.7+1
            p_score = modelD.forward(tensorp)
             #loss_real + loss_fake#
            
            # try discriminant on both distributions
            tensorq = torch.Tensor(next(q))
            if cuda:
                tensorq = tensorq.cuda(gpu)
            
            #tensorq = Variable(tensorq, requires_grad=True)
            q_score = modelD.forward(tensorq)
           
            loss_D_p = torch.mean(torch.log(p_score)) 
            loss_D_q = torch.mean(torch.log(torch.sub(1, q_score))) # zeros = fake
            lg2 = torch.log(torch.Tensor([2])) 
            if cuda:
                lg2 = lg2.cuda(gpu)
            loss_D = -1*(lg2 + 0.5*loss_D_p+ 0.5*loss_D_q)
            
            loss_D.backward()
            
            optimizerD.step()
        
        print(loss_D, iter)
            #loss_real = criterion(realD,torch.ones_like(realD))
    
 
    
    return loss_D
     
 
if __name__ == '__main__': 
    JS = []
    phi = np.linspace(-1, 1, 21)
    for i in phi:
        print(i)
        JS.append (JS_estimator(100,1,1,512,400,20, i))
  

