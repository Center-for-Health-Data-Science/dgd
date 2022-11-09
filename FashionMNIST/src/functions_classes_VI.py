import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model import SELayer, FmnistDGD

class VAE_Representation(torch.nn.Module):
    def __init__(self, 
                 nrep,        # Dimension of representation
                 nsample,     # Number of training samples
                 init=(0.,1.),
                 var_type='isotropic',
                 values=None
                 ):
        super(VAE_Representation, self).__init__()

        self.dz = None
        self.ds = None
        self.var_type = var_type
        self.mean = init[0]
        self.stddev = init[1]
        if values is None:
            self.nrep=nrep
            self.nsample=nsample
            self.mean, self.stddev = init[0],init[1]
            self.init_random(self.mean,self.stddev)
        else:
            # Initialize representations from a tensor with values
            self.nrep = values.shape[-1]
            self.nsample = values.shape[0]
            self.mean, self.stddev = None, self.stddev
            # Is this the way to copy values to a parameter?
            self.z = torch.nn.Parameter(torch.zeros_like(values), requires_grad=True)
            with torch.no_grad():
                self.z.copy_(values)
            if self.var_type == 'diagonal':
                self.s = torch.nn.Parameter(torch.full(fill_value=self.stddev, size=(self.nsample,self.nrep)), requires_grad=True)
            else:
                self.s = torch.nn.Parameter(torch.full(fill_value=self.stddev, size=(self.nsample,1)), requires_grad=True)
        
        self.pi_term = - 0.5*self.nrep*math.log(2*math.pi)
        self.betafactor = nrep*0.5
    
    def init_random(self,mean,stddev):
        self.z = torch.nn.Parameter(torch.normal(mean,stddev,size=(self.nsample,self.nrep), requires_grad=True))
        if self.var_type == 'diagonal':
            self.s = torch.nn.Parameter(torch.full(fill_value=self.stddev, size=(self.nsample,self.nrep)), requires_grad=True)
            #self.s = torch.nn.Parameter(torch.tensor([self.nsample,self.nrep]), requires_grad=True)
        else:
            print('isotropic')
            self.s = torch.nn.Parameter(torch.full(fill_value=self.stddev, size=(self.nsample,1)), requires_grad=True)
            #self.s = torch.nn.Parameter(torch.tensor([self.nsample,1]), requires_grad=True)
        #self.s.copy_(self.stddev)
    
    def forward(self, idx=None):
        if idx is None:
            return self.z, self.s
        else:
            return self.z[idx], self.s[idx]
    
    def reparametrize(self, idx=None):
        noise = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        if idx is None:
            return self.z + noise.sample(torch.Size([self.nsample])).to(self.z.device) * torch.exp(-0.5*self.s)
        else:
            return self.z[idx] + noise.sample(torch.Size([len(idx)])).to(self.z.device) * torch.exp(-0.5*self.s)[idx]
    
    def zero_grad(self):  # Used only if the update function is used
        if self.z.grad is not None:
            self.z.grad.detach_()
            self.z.grad.zero_()
        if self.s.grad is not None:
            self.s.grad.detach_()
            self.s.grad.zero_()

    def update(self,idx=None,lr=0.001,mom=0.9,wd=None):
        if self.dz is None:
            self.dz = torch.zeros(self.z.size()).to(self.z.device)
            self.ds = torch.zeros(self.s.size()).to(self.s.device)
        with torch.no_grad():
            self.dz[idx] = self.dz[idx].mul(mom) - self.z.grad[idx].mul(lr)
            self.ds[idx] = self.ds[idx].mul(mom) - self.s.grad[idx].mul(lr)
            if wd is not None:
                self.dz[idx] -= wd*self.z[idx]
                self.ds[idx] -= wd*self.s[idx]
            self.z[idx] += self.dz[idx]
            self.s[idx] += self.ds[idx]

###############################
# classes needed for VAE
###############################

class SimpleBlock_Enc(nn.Module):
    def __init__(self, c_in, c_out, k, s, p, activ='relu'):
        super(SimpleBlock_Enc, self).__init__()

        self.activ = activ

        if activ == 'relu':
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k, stride=s, padding=p),
                nn.ReLU(True)
            )
        elif activ == 'silu':
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k, stride=s, padding=p),
                nn.SiLU(True)
            )
        else:
            print('Error: no valid activation for FlexBlock specified')

        self.se_block = SELayer(c_out)

    def forward(self, x):
        x2 = self.block(x)
        x2 = self.se_block(x2)
        return x2

class Encoder(nn.Module):
    def __init__(self, capacity=32, activ='silu',
    skip = False, dropout=0, out_size=28, pixel_layers=(5), pixel_kernels=(5)):
        super(Encoder, self).__init__()

        self.flat = capacity * 3 * 3
        self.in_channels = 1
        self.capacity = capacity
        self.skip = skip
        self.activ = activ
        self.out_size = out_size
        
        self.block1 = SimpleBlock_Enc(c_in=self.in_channels,c_out=capacity,k=4,s=2,p=1,activ=activ)
        if self.skip:
            self.skip1 = SimpleBlock_Enc(c_in=self.in_channels,c_out=capacity*2,k=6,s=4,p=1,activ=activ)
        self.block2 = SimpleBlock_Enc(c_in=capacity,c_out=capacity*2,k=4,s=2,p=1,activ=activ)
        self.block3 = SimpleBlock_Enc(c_in=capacity*2,c_out=capacity*4,k=4,s=2,p=1,activ=activ)
        self.block4 = SimpleBlock_Enc(c_in=capacity*4,c_out=capacity,k=1,s=1,p=0,activ=activ)
        if self.skip:
            self.skip2 = SimpleBlock_Enc(c_in=capacity,c_out=capacity*4,k=6,s=4,p=1,activ=activ)

        
    def forward(self, x):
        if self.activ == 'silu':
            x1 = F.silu(self.block1(x))
            if self.skip:
                x2 = F.silu(self.block2(x1) + self.skip1(x))
                x3 = F.silu(self.block3(x2) + self.skip2(x1))
            else:
                x2 = F.silu(self.block2(x1))
                x3 = F.silu(self.block3(x2))
            x4 = F.silu(self.block4(x3))
            z = x4.view(x4.size(0), self.flat)
        else:
            x1 = F.relu(self.block1(x))
            if self.skip:
                x2 = F.relu(self.block2(x1) + self.skip1(x))
                x3 = F.relu(self.block3(x2) + self.skip2(x1))
            else:
                x2 = F.relu(self.block2(x1))
                x3 = F.relu(self.block3(x2))
            x4 = F.relu(self.block4(x3))
            z = x4.view(x4.size(0), self.flat)
        return z

class VAE(nn.Module):
    def __init__(self, latent=100, hidden=True, hidden_dim=256, capacity=64, activ='relu', pixel=True,
    skip = False, out_size=28, pixel_layers=(5),pixel_kernels=(5)):
        super(VAE, self).__init__()

        self.capacity = capacity
        self.latent = latent
        self.skip = skip
        self.activ = activ
        self.out_size = out_size

        self.encoder = FmnistDGD(capacity=capacity, activ=activ, skip=skip, out_size=out_size)

        if hidden:
            if self.activ == 'silu':
                self.enc1 = nn.Sequential(
                    nn.Linear(self.encoder.flat, hidden_dim),
                    nn.SiLU(True),
                    nn.Linear(hidden_dim, self.latent)
                )
                self.enc2 = nn.Sequential(
                    nn.Linear(self.encoder.flat, hidden_dim),
                    nn.SiLU(True),
                    nn.Linear(hidden_dim, self.latent)
                )
            else:
                self.enc1 = nn.Sequential(
                    nn.Linear(self.encoder.flat, hidden_dim),
                    nn.ReLU(True),
                    nn.Linear(hidden_dim, self.latent)
                )
                self.enc2 = nn.Sequential(
                    nn.Linear(self.encoder.flat, hidden_dim),
                    nn.ReLU(True),
                    nn.Linear(hidden_dim, self.latent)
                )
        else:
            self.enc1 = nn.Sequential(
                nn.Linear(self.encoder.flat, self.latent)
            )
            self.enc2 = nn.Sequential(
                nn.Linear(self.encoder.flat, self.latent)
            )
        
        self.decoder = Encoder(latent=latent,hidden=hidden,hidden_dim=hidden_dim,capacity=capacity,
                           activ=activ,pixel=pixel,skip=skip,out_size=out_size,
                           pixel_layers=pixel_layers,pixel_kernels=pixel_kernels)
        
    def reparametrize(self, m_u, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = m_u + (eps*std)
        return sample
    
    def forward(self, x):
        x = self.encoder(x)
        mu = self.enc1(x)
        logvar = self.enc2(x)

        z = self.reparametrize(mu, logvar)
        z = z.view(z.size(0), self.latent)
        y = self.decoder(z)

        return y, mu, logvar