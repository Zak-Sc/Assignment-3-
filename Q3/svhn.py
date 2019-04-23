
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.utils as utls
from classify_svhn import get_data_loader
import torchvision.utils as vutils
import torch.nn.functional as F


import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.autograd as autogd
from torch.autograd import Variable 
from torchsummary import summary

root ='C:/Users/binoS/Documents/ift6135/svhn/train_32x32.mat'

if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda") 
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)
class Generator(nn.Module):

    def __init__(self,image_shape=(3, 32, 32), noise_dim=128, dim_factor=64):
        super(Generator, self).__init__()
 
        C, H, W = image_shape
        assert H % 2**3 == 0, "Image height %i not compatible with architecture" % H
        self.H_init = int(H / 2**3)  # divide by 2^3 bc 3 deconvs with stride 0.5
        assert W % 2**3 == 0, "Image width %i not compatible with architecture" % W
        self.W_init = int(W / 2**3)  # divide by 2^3 bc 3 deconvs with stride 0.5

        self.linear = nn.Linear(noise_dim,
                                4 * dim_factor * self.H_init * self.W_init)
        self.deconv1 = nn.ConvTranspose2d(4 * dim_factor, 2 * dim_factor,
                                          4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(2 * dim_factor, dim_factor,
                                          4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(dim_factor, C,
                                          4, stride=2, padding=1)

    def forward(self, input):

     
        H1 = F.relu(self.linear(input))
        H1_resh = H1.view(H1.size(0), -1, self.W_init, self.H_init)
        H2 = F.relu(self.deconv1(H1_resh))
        H3 = F.relu(self.deconv2(H2))
        output = torch.tanh(self.deconv3(H3))
        return output
       
class    Discriminator(nn.Module):

    def __init__(self, image_shape=(3, 32, 32), dim_factor=64):# batch_size):
        super(Discriminator, self).__init__()
      
        C, H, W = image_shape
        assert H % 2**3 == 0, "Image height %i not compatible with architecture" % H
        H_out = int(H / 2**3)  # divide by 2^3 bc 3 convs with stride 2
        assert W % 2**3 == 0, "Image width %i not compatible with architecture" % W
        W_out = int(W / 2**3)  # divide by 2^3 bc 3 convs with stride 2

        self.pad = nn.ZeroPad2d(2)
        self.conv1 = nn.Conv2d(C, dim_factor, 5, stride=2)
        self.conv2 = nn.Conv2d(dim_factor, 2 * dim_factor, 5,
                               stride=2)
        self.conv3 = nn.Conv2d(2 * dim_factor, 4 * dim_factor, 5,
                               stride=2)
        self.linear = nn.Linear(4 * dim_factor * H_out * W_out, 1)

    def forward(self, input):
       
        H1 = F.leaky_relu(self.conv1(self.pad(input)), negative_slope=0.2)
        H2 = F.leaky_relu(self.conv2(self.pad(H1)), negative_slope=0.2)
        H3 = F.leaky_relu(self.conv3(self.pad(H2)), negative_slope=0.2)
        H3_resh = H3.view(H3.size(0), -1)  # reshape for linear layer
        output = self.linear(H3_resh)

        return output

    
def GP(modelD, real_image, fake_image, batch_size, Lambda):
     
    alpha = torch.rand(batch_size,1,1,1).to(device)
    alpha = alpha.expand(real_image.size())
    
    interp = (torch.mul(alpha , real_image) + torch.mul(torch.sub(1,alpha) , fake_image)).to(device)
    interp.requires_grad_()
    score_interp = modelD(interp)


    grad = autogd.grad(outputs=score_interp, inputs=interp,
                              grad_outputs=torch.ones(score_interp.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    penalty = torch.mean((grad.norm(2, dim=1) - 1) ** 2)
    return penalty


    
def evaluate(modelD, modelG, x, batch_size, hidden_size, Lambda, D_G):
    
    if D_G == 'D':
          
        real_image = x.to(device)
        real_score = modelD.forward(real_image)
                
        noise = torch.randn(batch_size, hidden_size).to(device)
        fake_image = modelG.forward(noise)
        fake_score = modelD.forward(fake_image)
    
        gradient_penalty = GP(modelD, real_image, fake_image, batch_size, Lambda)
        
        return -1*(torch.mean(real_score) - torch.mean(fake_score) - gradient_penalty*Lambda)
                
    if D_G == 'G':
                  
        noise = torch.randn(batch_size, hidden_size)
        fake = torch.Tensor(noise).to(device)
        fake_image = modelG.forward(fake)
        fake_score = modelD.forward(fake_image)
                    
        return -1* torch.mean(fake_score)
 

       
def enable_grad(model, enable):
    if enable:
        for p in model.parameters(): 
            p.requires_grad = True  
    else:
        for p in model.parameters(): 
            p.requires_grad = False  



def generate_image(model, hidden_size,iter):
        noise = torch.randn(64, hidden_size).to(device)
        with torch.no_grad():
            test_image = model.forward(noise).detach().cpu()
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(test_image, nrow=10, padding=2, normalize=True), (1,2,0)))
        plt.savefig("Epoch_%d {}".format('svhn') %(iter+1))
        plt.close('all')           
 
    
    
def training(train, epoch, BATCHSIZE, hidden_size,Lambda):

    modelD = Discriminator((3,32,32),BATCHSIZE).to(device)
    modelG = Generator((3, 32, 32),hidden_size,BATCHSIZE).to(device)

    optimizerD = optim.Adam(modelD.parameters(), lr=1e-3, betas=(0.3, 0.9))
    optimizerG = optim.Adam(modelG.parameters(), lr=1e-3, betas=(0.3, 0.9))
    
 
    for iter in range(epoch):
         avg_lossD = 0
         avg_lossG = 0
         for i, (x, y) in enumerate(train):
            
            if x.shape[0] == BATCHSIZE: 
                
                enable_grad(modelD,True)
                enable_grad(modelG,False)
      
                optimizerD.zero_grad()  
                
                loss_D = evaluate(modelD, modelG, x, BATCHSIZE, hidden_size, Lambda, 'D')
                
                avg_lossD += (loss_D)
  
                loss_D.backward()
  
                optimizerD.step()
                
        
                if (i+1)%3==0:  # update G every 3 loops
                    
                    modelG.train()
                    enable_grad(modelG,True)
                    enable_grad(modelD,False)
                    optimizerG.zero_grad()
                    
                    loss_G = evaluate(modelD, modelG, x, BATCHSIZE, hidden_size, Lambda, 'G')
                    avg_lossG += (loss_G)
                    loss_G.backward()
                    optimizerG.step()
                    
                 
                if (i+1)%320 == 0:
                   
                     print(" [NLL] iter {} / Disc {} / Gen {}".format( 
                                 iter,(avg_lossD)/100,(avg_lossG)/100))
                     avg_lossD = 0
                     avg_lossG = 0
        
         if(iter+1)%10==0:  
             generate_image(modelG, hidden_size,iter)
                
                           
    return modelG
   
        
 
if __name__ == '__main__': 
    train, valid, test = get_data_loader("svhn", 32)
                            #120 epochs 32 batchsize 100 hiddensize 10 lambda
    Gen= training(train, 120, 32, 100, 10 )
  
    

