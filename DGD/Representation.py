import math
import torch
import torch.nn as nn
import torch.distributions as D

class RepresentationLayer(torch.nn.Module):
    '''
    Implements a representation layer, that accumulates pytorch gradients.

    Representations are vectors in nrep-dimensional real space. By default
    they will be initialized as a tensor of dimension nsample x nrep from a
    normal distribution (mean and variance given by init).

    One can also supply a tensor to initialize the representations (values=tensor).
    The representations will then have the same dimension and will assumes that
    the first dimension is nsample (and the last is nrep).

    forward() takes a sample index and returns the representation.

    Representations are "linear", so a representation layer may be followed
    by an activation function.
    '''
    def __init__(self,
                nrep,        # Dimension of representation
                nsample,     # Number of training samples
                init=(0.,1.),# Normal distribution mean and stddev for
                                # initializing representations
                values=None  # If values is given, the other parameters are ignored
                ):
        super(RepresentationLayer, self).__init__()
        self.dz = None
        if values is None:
            self.nrep=nrep
            self.nsample=nsample
            self.mean, self.stddev = init[0],init[1]
            self.init_random(self.mean,self.stddev)
        else:
            # Initialize representations from a tensor with values
            self.nrep = values.shape[-1]
            self.nsample = values.shape[0]
            self.mean, self.stddev = None, None
            # Is this the way to copy values to a parameter?
            self.z = torch.nn.Parameter(torch.zeros_like(values), requires_grad=True)
            with torch.no_grad():
                self.z.copy_(values)

    def init_random(self,mean,stddev):
        # Generate random representations
        self.z = torch.nn.Parameter(torch.normal(mean,stddev,size=(self.nsample,self.nrep), requires_grad=True))

    def forward(self, idx=None):
        if idx is None:
            return self.z
        else:
            return self.z[idx]
    
    def __getitem__(self,idx):
        return self.z[idx]

    def rescale(self):
        z_flat = torch.flatten(self.z.cpu().detach())
        sd, m = torch.std_mean(z_flat)
        with torch.no_grad():
            self.z -= m
            self.z /= sd