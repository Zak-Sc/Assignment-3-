# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 20:05:27 2019

@author: binoS
"""

# -*- coding: utf-8 -*-
"""

Created on Thu Apr 18 17:09:23 2019

@author: binoS
"""

import numpy as np
import torch 

import matplotlib.pyplot as plt
import torch.nn as nn 
import torch.optim as optim
import torch.autograd as autogd
from torch.autograd import Variable 
import samplers
from torchsummary import summary

cuda = torch.cuda.is_available()
if cuda:
    gpu = 0

class Critic(nn.Module):

    def __init__(self,input_shape, hidden_shape):
        super(Critic, self).__init__()

        
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

    
def GP(modelD, realD, fakeD, batch_size, Lambda):
    alpha = torch.rand(batch_size, 1)
    if cuda:
        alpha = alpha.cuda(gpu)
    interp = torch.mul(alpha , realD) + torch.mul(torch.sub(1,alpha) , fakeD)
    if cuda:
        interp = interp.cuda(gpu)
        
    score_interp = modelD(interp)

    # TODO: Make ConvBackward diffentiable
    grad = autogd.grad(outputs=score_interp, inputs=interp,
                              grad_outputs=torch.ones(score_interp.size()).cuda(gpu) if cuda else torch.ones(
                                  score_interp.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    penalty = torch.mean((grad.norm(2, dim=1) - 1) ** 2)
    return penalty
    
def Wasserstein(epoch, choice_p,choice_q,BATCHSIZE, hidden_size, d_iter,Lambda, phi ):

    modelD = Critic(2,hidden_size)
    if cuda:
        modelD = modelD.cuda(gpu)
    
    optimizerD = optim.SGD(modelD.parameters(), lr=1e-3) # ,weight_decay=0.001
    
    p = dist(choice_p,BATCHSIZE,0)
    q = dist(choice_q,BATCHSIZE,phi)
    
    for iter in range(epoch):
        
        for d_it in range(d_iter):
            modelD = modelD.train()
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
           
            #gradient_penalty = GP(modelD, tensorp, tensorq,BATCHSIZE, Lambda)
            alpha = torch.rand(BATCHSIZE, 1)
            if cuda:
                alpha = alpha.cuda(gpu)
            interp = Variable(torch.mul(alpha , tensorp) + torch.mul(torch.sub(1,alpha) , tensorq),requires_grad=True)
            if cuda:
                interp = interp.cuda(gpu)
                
            score_interp = modelD(interp)
            
            # TODO: Make ConvBackward diffentiable
            grad = autogd.grad(outputs=score_interp, inputs=interp,
                                      grad_outputs=torch.ones(score_interp.size()).cuda(gpu) if cuda else torch.ones(
                                          score_interp.size()),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
        
            gradient_penalty = torch.mean((grad.norm(2, dim=1) - 1) ** 2)
                    #gradient_penalty.backward()         
            
            loss_D = -1*(torch.mean(p_score) - torch.mean(q_score) - gradient_penalty*Lambda)
            loss_D.backward()
            
       
            optimizerD.step()
         

        
       
    print(loss_D)
        
    return loss_D
 
if __name__ == '__main__': 
    WT = []
    phi = np.linspace(-1, 1, 21)
    for i in phi:
        print(i)
        WT.append(Wasserstein(100,1,1,512,200,100,10,i))
    

