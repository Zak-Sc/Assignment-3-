from torchvision.datasets import utils
import torch.utils.data as data_utils
import torch
import os
import numpy as np
from torch import nn
from torch.nn.modules import upsampling
from torch.functional import F
from torch.optim import Adam
from torchsummary import summary
import torch.optim as optim
def get_data_loader(dataset_location, batch_size):
    URL = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
    # start processing
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    splitdata = []
    for splitname in ["train", "valid", "test"]:
        filename = "binarized_mnist_%s.amat" % splitname
        filepath = os.path.join(dataset_location, filename)
        utils.download_url(URL + filename, dataset_location)
        with open(filepath) as f:
            lines = f.readlines()
        x = lines_to_np_array(lines).astype('float32')
        x = x.reshape(x.shape[0], 1, 28, 28)
        # pytorch data loader
        dataset = data_utils.TensorDataset(torch.from_numpy(x))
        dataset_loader = data_utils.DataLoader(x, batch_size=batch_size, shuffle=splitname == "train")
        splitdata.append(dataset_loader)
    return splitdata
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)
#Hyperparameters
batch_size = 64
n_epochs = 20
K=200
log_interval = 100
lr = 3e-4
latent_size = 100
input_size=(1,28,28)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        self.pool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(5, 5))
        self._mean = nn.Linear(in_features=256, out_features=100, bias=True)
        self._logvar = nn.Linear(in_features=256, out_features=100, bias=True)
        self.elu = nn.ELU()

    def forward(self, input):
        input = self.conv1(input)
        input = self.elu(input)
        input = self.pool1(input)
        input = self.conv2(input)
        input = self.elu(input)
        input = self.pool2(input)
        input = self.conv3(input)
        input = self.elu(input)
        input = input.view(input.size(0), -1)
        mean = self._mean(input)
        logvar = self._logvar(input)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(in_features=100, out_features=256, bias=True)
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(5, 5), padding=4)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=2)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=2)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(3, 3), padding=2)
        self.elu = nn.ELU()

    def forward(self, input):
        input = self.linear(input)
        input = input.reshape(input.size(0), 256, 1, 1)
        input = self.elu(input)
        input = self.conv1(input)
        input = self.elu(input)
        input = self.upsample1(input)
        input = self.conv2(input)
        input = self.elu(input)
        input = self.upsample2(input)
        input = self.conv3(input)
        input = self.elu(input)
        return torch.sigmoid(self.conv4(input))

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input):
        mean, logvar = self.encoder(input)
        epsilon = torch.randn_like(logvar)
        input = mean + torch.exp(logvar / 2) * epsilon 
        return self.decoder(input), mean, logvar
model = VAE()
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
summary(model, input_size=input_size)
#src=https://github.com/pytorch/examples/blob/master/vae/main.py
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())

    return BCE + KLD
#src=https://github.com/pytorch/examples/blob/master/vae/main.py
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(x_train):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(x_train.dataset),
                100. * batch_idx / len(x_train),
                -loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, -train_loss / len(x_train.dataset)))
for epoch in range(n_epochs):
        train(epoch)
model.eval()
loss = 0
for batch_idx, data in enumerate(x_valid):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss += loss_function(recon_batch, data, mu, logvar).item()
loss = loss/len(x_valid.dataset)
print('Valid - average per-instance ELBO: {} '.format(-loss))
model.eval()
loss = 0
losses=[]
for batch_idx, data in enumerate(x_test):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss += loss_function(recon_batch, data, mu, logvar).item()
            losses.append(loss)
loss = loss/len(x_valid.dataset)
print('Test - average per-instance ELBO: {} '.format(-loss))
#param1: K  the number of importance samples
#param2: mean
#param3: std
#return importance samples of size (M,K,L)
def get_samples(K, mu, std):
    mu = mu.unsqueeze(1).expand(-1, K, -1) 
    std = std.unsqueeze(1).expand(-1, K, -1)    
    return torch.normal(mu, std)
#Evaluating log-likelihood with Variational Autoencoders 
#param1: trained  model
#param2: tensor x of data (M,D)
#param3: tensor z_samples ( M, K, L)
#return (log p(x_), ..., log p(x_M))
def batch_log_px(model, x, z): 
        with torch.no_grad():
              M = x.shape[0]
              D = x.shape[1]
              K = z.shape[1]
              L = z.shape[2]
              mu, logvar = model.encoder(x.reshape((M, 1,28,28)))
              std = torch.exp(logvar / 2)
              mu = mu.unsqueeze(1).expand(-1, K, -1) 
              std = std.unsqueeze(1).expand(-1, K, -1)
              #q(z|x)
              qz_x=(1/torch.sqrt(2*math.pi*std**2))*torch.exp(((z - mu)/std)**2)
              log_qz_x=torch.sum(torch.log(qz_x), -1)
              #p(z)
              pz=(1/math.sqrt(2*math.pi))*torch.exp(z**2)
              log_pz=torch.sum(torch.log(pz), -1)
              #p(x|z)
              recon_x = model.decoder(z.reshape((M*K, L)))
              recon_x= recon_x.reshape((M, K, D))
              log_px_z = torch.sum((x.unsqueeze(1).expand(-1, K, -1) * (recon_x).log() + (1 - x.unsqueeze(1).expand(-1, K, -1)) * (1 - recon_x).log()),-1)
              #log p(x) = log(1/K) + log(sum(exp(log p(x|z)+log p(z)-log q(z|x))))
              log_terms = log_px_z + log_pz - log_qz_x
              max_log,_= log_terms.max(dim=-1, keepdim=True)
              log_px=-np.log(K)+max_log+torch.log(torch.sum(torch.exp(log_terms-max_log)))
        return log_px.mean()
#return (log p(x_1), ..., log p(x_M))
def get_log_px_dataset(data_loader, K):
    log_px_all = []
    for i, x in enumerate(data_loader):
        x = x.to(device)
        mu, logvar = model.encoder(x)
        std = torch.exp(0.5*logvar)
        z = get_samples(K,mu,std)
        log_px_all.append(batch_log_px(model,x.reshape((x.shape[0], -1)),z).cpu().numpy())        
    return log_px_all
log_px_valid = get_log_px_dataset(x_valid, 200)
print('Validation -  log-likelihood: {} '.format(np.mean(log_px_valid)))
print('Test -  log-likelihood: {} '.format(np.mean(log_px_test)))