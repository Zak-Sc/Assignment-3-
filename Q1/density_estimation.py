#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""

from __future__ import print_function
import numpy as np
import torch 

import matplotlib.pyplot as plt
import torch.nn as nn 
import torch.optim as optim 
import samplers
from torch.autograd import Variable 


# plot p0 and p1
plt.figure()

# empirical
xx = torch.randn(10000)
f = lambda x: torch.tanh(x*2+1) + x*0.75
d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
plt.hist(f(xx), 100, alpha=0.5, density=1)
plt.hist(xx, 100, alpha=0.5, density=1)
plt.xlim(-5,5)
# exact
xx = np.linspace(-5,5,1000)
N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.plot(xx, N(xx))


############### import the sampler ``samplers.distribution4'' 
############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######
 


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
    if choice ==2:
        dist = iter(samplers.distribution2(size)) 
    if choice ==3:
        dist = iter(samplers.distribution3(size)) 
        return dist 
    if choice ==4:
        dist = iter(samplers.distribution4(size)) 
        return dist 
    
def Density_estimator(epoch,BATCHSIZE, hidden_size, d_iter):

    modelD = Discriminant(1,hidden_size)
    if cuda:
        modelD = modelD.cuda(gpu)
        
    optimizerD = optim.SGD(modelD.parameters(), lr=1e-2) # ,weight_decay=0.001
    #nn.CrossEntropyLoss()
    p = dist(4,BATCHSIZE,0)
    q = dist(3,BATCHSIZE,0)
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
          
            loss_D = -1*(loss_D_p + loss_D_q)
            
            loss_D.backward()
            
            optimizerD.step()

        
    print(loss_D, iter)
            #loss_real = criterion(realD,torch.ones_like(realD))
    

    return modelD


trained_model = Density_estimator(100,512,1000,20)
    

############### plotting things
############### (1) plot the output of your trained discriminator 
############### (2) plot the estimated density contrasted with the true density

ll = xx.reshape(len(xx),1)
ndata = torch.Tensor(ll)
rrr = trained_model.forward(ndata) # evaluate xx using your discriminator; replace xx with the output
rrr = rrr.detach().numpy().flatten()
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(xx,rrr)
plt.title(r'$D(x)$')

estimate = N(xx)*(rrr/(1-rrr))# estimate the density of distribution4 (on xx) using the discriminator; 
                                # replace "np.ones_like(xx)*0." with your estimate
plt.subplot(1,2,2)
plt.plot(xx,estimate)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.legend(['Estimated','True'])
plt.title('Estimated vs True')











