import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, dropout=0):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if dropout != 0:
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction, bias=False),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel, bias=False),
                nn.Dropout(dropout),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SimpleBlock(nn.Module):
    def __init__(self, c_in, c_out, k, s, p, activ='relu'):
        super(SimpleBlock, self).__init__()

        self.activ = activ

        if activ == 'relu':
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels=c_in, out_channels=c_out, kernel_size=k, stride=s, padding=p),
                nn.ReLU(True)
            )
        elif activ == 'silu':
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels=c_in, out_channels=c_out, kernel_size=k, stride=s, padding=p),
                nn.SiLU(True)
            )
        else:
            print('Error: no valid activation for FlexBlock specified')

        self.se_block = SELayer(c_out)

    def forward(self, x):
        x2 = self.block(x)
        x2 = self.se_block(x2)
        return x2

class MaskedCNN(nn.Conv2d):
    """
	Implementation of Masked CNN Class as explained in A Oord et. al. 
	Taken from https://github.com/jzbontar/pixelcnn-pytorch
	"""
    
    # arguments are mask_type, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
    def __init__(self, mask_type, *args, **kwargs):
        self.mask_type = mask_type
        assert mask_type in ['A', 'B'], "Unknown Mask Type"
        super(MaskedCNN, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())

        _, depth, height, width = self.weight.size()
        self.mask.fill_(1)
        if mask_type =='A':
            self.mask[:,:,height//2,width//2:] = 0
            self.mask[:,:,height//2+1:,:] = 0
        else:
            self.mask[:,:,height//2,width//2+1:] = 0
            self.mask[:,:,height//2+1:,:] = 0


    def forward(self, x):
        self.weight.data*=self.mask
        return super(MaskedCNN, self).forward(x)

class PixelCNN(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel=7, channels=64, n_layers=5, activ='relu'):
        super(PixelCNN, self).__init__()

        self.main = nn.ModuleList()

        self.main.append(MaskedCNN('A', in_channels=in_channels, out_channels=channels, kernel_size=kernel, stride=1, padding=kernel//2, bias=False))
        for i in range(1,n_layers):
            self.main.append(MaskedCNN('B', in_channels=channels, out_channels=channels, kernel_size=kernel, stride=1, padding=kernel//2, bias=False))
            if activ == 'silu':
                self.main.append(nn.SiLU(True))
            else:
                self.main.append(nn.ReLU(True))
        self.main.append(nn.Conv2d(in_channels=channels, out_channels=out_channels, kernel_size=1))
        if activ == 'silu':
            self.main.append(nn.SiLU(True))
        else:
            self.main.append(nn.ReLU(True))
    
    def forward(self, x):
        for i in range(len(self.main)):
            x = self.main[i](x)
        return x

class FmnistDGD(nn.Module):
    def __init__(self, latent=50, hidden=True, hidden_dim=256, capacity=64, activ='relu', pixel=True,
    skip = True, dropout=0, pixel_layers=(5), pixel_kernels=(5)):
        super(FmnistDGD, self).__init__()

        self.latent = latent
        self.flat = capacity * 3 * 3
        self.out_channels = 1
        self.capacity = capacity
        self.skip = skip
        self.activ = activ

        if type(hidden_dim) is not int:
            n_hidden = len(hidden_dim)
        else:
            n_hidden = 1

        self.init_fc = nn.ModuleList()
        if hidden:
            if n_hidden > 1:
                for i in range(n_hidden):
                    if i == 0:
                        self.init_fc.append(nn.Linear(latent,hidden_dim[0]))
                    else:
                        self.init_fc.append(nn.Linear(hidden_dim[i-1],hidden_dim[i]))
                    if dropout != 0:
                        self.init_fc.append(nn.Dropout(dropout))
                    if activ == 'silu':
                        self.init_fc.append(nn.SiLU(True))
                    else:
                        self.init_fc.append(nn.ReLU(True))
                self.init_fc.append(nn.Linear(hidden_dim[-1],self.flat))
                if dropout != 0:
                    self.init_fc.append(nn.Dropout(dropout))
                if activ == 'silu':
                    self.init_fc.append(nn.SiLU(True))
                else:
                    self.init_fc.append(nn.ReLU(True))
            else:
                self.init_fc.append(nn.Linear(latent,hidden_dim))
                if dropout != 0:
                    self.init_fc.append(nn.Dropout(dropout))
                if activ == 'silu':
                    self.init_fc.append(nn.SiLU(True))
                else:
                    self.init_fc.append(nn.ReLU(True))
                self.init_fc.append(nn.Linear(hidden_dim,self.flat))
                if dropout != 0:
                    self.init_fc.append(nn.Dropout(dropout))
                if activ == 'silu':
                    self.init_fc.append(nn.SiLU(True))
                else:
                    self.init_fc.append(nn.ReLU(True))
        else:
            self.init_fc.append(nn.Linear(latent,self.flat))
            if dropout != 0:
                self.init_fc.append(nn.Dropout(dropout))
            if activ == 'silu':
                self.init_fc.append(nn.SiLU(True))
            else:
                self.init_fc.append(nn.ReLU(True))
        
        self.block1 = SimpleBlock(c_in=capacity,c_out=capacity*8,k=1,s=1,p=0,activ=activ)
        self.block2 = SimpleBlock(c_in=capacity*8,c_out=capacity*4,k=4,s=2,p=1,activ=activ)
        self.block3 = SimpleBlock(c_in=capacity*4,c_out=capacity*2,k=4,s=2,p=0,activ=activ)
        if self.skip:
            self.skip2 = SimpleBlock(c_in=capacity*8,c_out=capacity*2,k=6,s=4,p=0,activ=activ)
        self.block4 = SimpleBlock(c_in=capacity*2,c_out=capacity,k=4,s=2,p=1,activ=activ)
        if self.skip:
            self.skip1 = SimpleBlock(c_in=capacity,c_out=capacity*4,k=4,s=2,p=1,activ=activ)
        
        self.pixel = pixel
        if self.pixel:
            if type(pixel_layers) is not int:
                self.pixel1 = PixelCNN(in_channels=capacity*2,out_channels=capacity*2,channels=capacity*2,kernel=pixel_kernels[0],n_layers=pixel_layers[0],activ=self.activ)
                self.pixel2 = PixelCNN(in_channels=capacity,out_channels=capacity,channels=capacity,kernel=pixel_kernels[1],n_layers=pixel_layers[1],activ=self.activ)
            else:
                self.pixel1 = None
                self.pixel2 = PixelCNN(in_channels=capacity,out_channels=capacity,channels=capacity,kernel=pixel_kernels,n_layers=pixel_layers,activ=self.activ)
        
        self.block_out = nn.Sequential(
            nn.ConvTranspose2d(in_channels=capacity, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        for i in range(len(self.init_fc)):
            z = self.init_fc[i](z)
        z = z.view(z.size(0), self.capacity, 3, 3)
        if self.activ == 'silu':
            x1 = F.silu(self.block1(z))
        else:
            x1 = F.relu(self.block1(z))
        if self.skip:
            if self.activ == 'silu':
                x2 = F.silu(self.block2(x1) + self.skip1(z), inplace=True)
                x3 = F.silu(self.block3(x2) + self.skip2(x1), inplace=True)
            else:
                x2 = F.relu(self.block2(x1) + self.skip1(z), inplace=True)
                x3 = F.relu(self.block3(x2) + self.skip2(x1), inplace=True)
            if self.pixel:
                if self.pixel1 is not None:
                    x3 = self.pixel1(x3)
            if self.activ == 'silu':
                x4 = F.silu(self.block4(x3), inplace=True)
            else:
                x4 = F.relu(self.block4(x3), inplace=True)
        else:
            if self.activ == 'silu':
                x2 = F.silu(self.block2(x1), inplace=True)
                x3 = F.silu(self.block3(x2), inplace=True)
            else:
                x2 = F.relu(self.block2(x1), inplace=True)
                x3 = F.relu(self.block3(x2), inplace=True)
            if self.pixel:
                if self.pixel1 is not None:
                    x3 = self.pixel1(x3)
            if self.activ == 'silu':
                x4 = F.silu(self.block4(x3), inplace=True)
            else:
                x4 = F.relu(self.block4(x3), inplace=True)
        if self.pixel:
            x4 = self.pixel2(x4)
        x_out = self.block_out(x4)
        return x_out