import os
import torch
import torchvision
import numpy as np
import math
import torch.nn as nn
import torch.distributions as D
import pandas as pd

class FashionMNIST_Dataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        super(FashionMNIST_Dataset, self).__init__()
        self.data = torchvision.datasets.FashionMNIST(root='./data', train=train,
                                            download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
        self.nsample = len(self.data)

    def __getitem__(self, index):
        data, target = self.data[index]
        return data, target, index

    def __len__(self):
        return self.nsample

class SingleGaussian(nn.Module):
    def __init__(self, dim, init = (0.,1.)):
        '''
        Reduced version of GaussianMixture for single, fixed Gaussian (not trained)
        with mean 0 and sd 1
        '''
        super(SingleGaussian, self).__init__()
        self.mean = nn.Parameter(torch.tensor([init[0]]),requires_grad=False)
        self.sd = nn.Parameter(torch.tensor([init[1]]),requires_grad=False)
        self.dim = dim
        self.Nmix = 1

        self.pi_term = - 0.5 * self.dim * math.log(2 * math.pi)
  
    def forward(self,x):
        # taking the sd into account is not relevant for a Gaussian N(0,1)
        # but we account for flexibility of value for fixed sd
        beta = 1/(self.sd**2)
        x = self.pi_term - 0.5*(torch.log(self.sd) + (x.unsqueeze(-2)-self.mean).square().mul(beta)).sum(-1)
        return torch.abs(x)

    def Distribution(self,temp=1):
        with torch.no_grad():
            return D.Normal(self.mean,self.sd*temp)

    def sample(self,nsample,temp=1):
        with torch.no_grad():
            gmm = self.Distribution(temp)
            return gmm.sample(torch.tensor([nsample,self.dim])).squeeze(2)

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def kld(mu, logvar):
    kld_term = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kld_term

def train(model,gmm,rep,test_rep,data,testdata,latent,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    optim_beta=[0.5,0.7],wd=1e-4,nepochs=500,
    lr_schedule_epochs=[0],lr_schedule=[[1e-3,1e-2,1e-1]],start_epoch=0,
    input_dims=28*28,model_type='dgd',supervision_labels=None,
    export_dir=None,export_name=None,save_stuff=True):

    nsample = data.dataset.__len__()
    nsample_test = testdata.dataset.__len__()

    model=model.to(device)
    if model_type != 'vae':
        rep = rep.to(device)
        test_rep = test_rep.to(device)
    
    if gmm is not None:
        gmm = gmm.to(device)
        if gmm.mean.requires_grad:
            train_gmm = True
            lr_gmm = lr_schedule[0][2]
            gmm_optimizer = torch.optim.Adam(gmm.parameters(), lr=lr_gmm, weight_decay=wd,betas=(optim_beta[0],optim_beta[1]))
        else:
            train_gmm = False
    else:
        train_gmm = False
    
    lr_decoder = lr_schedule[0][0]
    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr_decoder, weight_decay=wd,betas=(optim_beta[0],optim_beta[1]))
    if model_type != 'vae':
        lr_rep = lr_schedule[0][1]
        rep_optimizer = torch.optim.Adam(rep.parameters(), lr=lr_rep, weight_decay=wd,betas=(optim_beta[0],optim_beta[1]))
        testrep_optimizer = torch.optim.Adam(test_rep.parameters(), lr=lr_rep, weight_decay=wd,betas=(optim_beta[0],optim_beta[1]))

    criterion = nn.BCELoss(reduction='sum')
    
    train_avg = []
    recon_avg = []
    test_avg = []
    recon_test_avg = []
    dist_avg = []
    dist_test_avg = []
    
    if export_name is not None:
        if not os.path.exists(export_dir+export_name):
            os.makedirs(export_dir+export_name)
    
    for epoch in range(start_epoch,nepochs):

        if lr_schedule_epochs is not None:
            if epoch in lr_schedule_epochs:
                lr_idx = [x for x in range(len(lr_schedule_epochs)) if lr_schedule_epochs[x] == epoch][0]
                lr_decoder = lr_schedule[lr_idx][0]
                model_optimizer = torch.optim.Adam(model.parameters(), lr=lr_decoder, weight_decay=wd, betas=(optim_beta[0],optim_beta[1]))
                lr_rep = lr_schedule[lr_idx][1]
                lr_gmm = lr_schedule[lr_idx][2]
                if gmm is not None:
                    gmm_optimizer = torch.optim.Adam(gmm.parameters(), lr=lr_gmm, weight_decay=wd,betas=(optim_beta[0],optim_beta[1]))

        train_avg.append(0)
        recon_avg.append(0)
        test_avg.append(0)
        recon_test_avg.append(0)
        dist_avg.append(0)
        dist_test_avg.append(0)
        
        # train
        model.train()
        if rep_optimizer is not None:
            rep_optimizer.zero_grad()
        if model_type != 'vae':
            rep.zero_grad()
        for x, t, i in data:
            if train_gmm:
                gmm_optimizer.zero_grad()
            model_optimizer.zero_grad()
            x = x.to(device)
            if model_type == 'vae':
                y, mu, logvar = model(x)
                kld_loss = kld(mu, logvar)
                recon_loss_x = criterion(y,x)
                loss = recon_loss_x.clone() + kld_loss.clone()
            elif model_type == 'vad':
                z = rep.reparametrize(i)
                mu = rep.z[i]
                logvar = rep.s[i]
                y = model(z)
                kld_loss = kld(mu, logvar)
                recon_loss_x = criterion(y,x)
                loss = recon_loss_x.clone() + kld_loss.clone()
            else:
                z = rep(i)
                y = model(z)
                recon_loss_x = criterion(y,x)
                if supervision_labels is not None:
                    sup_i = supervision_labels[i]
                    gmm_error = - gmm(z, sup_i).sum()
                else:
                    if gmm is not None:
                        gmm_error = - gmm(z).sum()
                if gmm is not None:
                    loss = recon_loss_x.clone() + gmm_error.clone()
                else:
                    loss = recon_loss_x.clone()
            loss.backward()

            # update parameters
            if model_type != 'vae':
                if rep_optimizer is None:
                    rep.update(i,lr=lr_rep,mom=0.9)
                    if (not train_gmm) and (model_type == 'dgd'):
                        rep.rescale()
                if train_gmm:
                    gmm_optimizer.step()
            model_optimizer.step()

            train_avg[-1] += loss.item()
            recon_avg[-1] += recon_loss_x.item()
            if model_type != 'dgd':
                dist_avg[-1] += kld_loss.item()
            else:
                if gmm is not None:
                    dist_avg[-1] += gmm_error.item()
        if rep_optimizer is not None:
            rep_optimizer.step()
        
        train_avg[-1] /= nsample*input_dims
        recon_avg[-1] /= nsample*input_dims
        if gmm is not None:
            dist_avg[-1] /= nsample*latent*gmm.Nmix
        else:
            dist_avg[-1] /= nsample*latent
        
        model.eval()
        if testrep_optimizer is not None:
            testrep_optimizer.zero_grad()
        if model_type == 'dgd':
            test_rep.zero_grad()
        #n_batches = 0
        for x, t, i in testdata:
            if train_gmm:
                gmm_optimizer.zero_grad()
            model_optimizer.zero_grad()
            x = x.to(device)
            if model_type == 'vae':
                y, mu, logvar = model(x)
                kld_loss = kld(mu, logvar)
                recon_loss_x = criterion(y,x)
                loss = recon_loss_x.clone() + kld_loss.clone()
            elif model_type == 'vad':
                z = test_rep.reparametrize(i)
                mu = test_rep.z[i]
                logvar = test_rep.s[i]
                y = model(z)
                kld_loss = kld(mu, logvar)
                recon_loss_x = criterion(y,x)
                loss = recon_loss_x.clone() + kld_loss.clone()
            else:
                # get representation from data indices
                z = test_rep(i)
                y = model(z)
                recon_loss_x = criterion(y,x)
                if gmm is not None:
                    gmm_error = - gmm(z).sum()
                    loss = recon_loss_x.clone() + gmm_error.clone()
                else:
                    loss = recon_loss_x.clone()
            loss.backward()

            # update parameters
            if model_type != 'vae':
                if testrep_optimizer is None:
                    test_rep.update(i,lr=lr_rep,mom=0.9)
                    if (not train_gmm) and (model_type == 'dgd'):
                        test_rep.rescale()
            
            test_avg[-1] += loss.item()
            recon_test_avg[-1] += recon_loss_x.item()
            if model_type != 'dgd':
                dist_test_avg[-1] += kld_loss.item()
            else:
                if gmm is not None:
                    dist_test_avg[-1] += gmm_error.item()
            #n_batches += 1
        
        if testrep_optimizer is not None:
            testrep_optimizer.step()
        
        test_avg[-1] /= nsample_test*input_dims
        recon_test_avg[-1] /= nsample_test*input_dims
        if gmm is not None:
            dist_test_avg[-1] /= nsample_test*latent*gmm.Nmix
        else:
            dist_test_avg[-1] /= nsample_test*latent
    
    history = pd.DataFrame({'epoch': np.arange(nepochs),
                            'loss': train_avg,
                            'Recon_loss': recon_avg,
                            'Pz_loss': dist_avg,
                            'type': 'train'
                            })
    temp = pd.DataFrame({'epoch': np.arange(nepochs),
                            'loss': test_avg,
                            'Recon_loss': recon_test_avg,
                            'Pz_loss': dist_test_avg,
                            'type': 'test'
                            }, index=[x+nepochs for x in np.arange(nepochs)])
    history = history.append(temp)

    if save_stuff:
        if model_type == 'vae':
            torch.save(model.state_dict(), export_dir+export_name+'/'+export_name+'_model.pt')
        else:
            torch.save(model.state_dict(), export_dir+export_name+'/'+export_name+'_decoder.pt')
            torch.save(rep.state_dict(), export_dir+export_name+'/'+export_name+'_representation.pt')
            torch.save(test_rep.state_dict(), export_dir+export_name+'/'+export_name+'_testRepresentation.pt')
        if train_gmm:
            torch.save(gmm.state_dict(), export_dir+export_name+'/'+export_name+'_gmm.pt')
        history.to_csv(export_dir+export_name+'/'+export_name+'_history.csv')
    
    return history, model, gmm, rep, test_rep

def supervision_list(t_set, ratio, classes):
    step = int(1 / ratio)
    labels = [classes[x[1]] for x in t_set.data]
    unique_classes = list(set(labels))
    id_position_out = np.zeros(len(labels))
    for lbl in unique_classes:
        ids = [x for x in range(len(labels)) if labels[x] == lbl]
        keep = [ids[x] for x in range(len(ids)) if x % step == 0]
        id_position_out[keep] = 1
    out = np.full(shape=(t_set.__len__()),fill_value=999)
    out[np.where(id_position_out == 1)[0]] = t_set.data.targets[np.where(id_position_out == 1)[0]]
    return out