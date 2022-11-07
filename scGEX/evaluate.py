import torch
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from DGD.Prior import GaussianMixture, softball
from src.model import DGD
from src.functions import scDataset
from DGD.Representation import RepresentationLayer
from src.functions import set_random_seed


seed = 0
set_random_seed(seed)

############################
# read in all data needed
############################

data_path = './data/'

# read original count matrix
count_mtx = scipy.io.mmread(data_path+'full_data.mtx')

# get meta data
with open(data_path+'meta.pkl', 'rb') as f:
    meta_data = pickle.load(f)
# get train-validation-test split for reproducibility
with open(data_path+'val_test_split.pkl', 'rb') as f:
    val_test_split = pickle.load(f)
# get gene names of transcript features
genes = pd.read_csv(data_path+'genes.tsv', sep='\t', header=None)
genes.columns = ['gene_ID', 'gene_name']

############################
# prepare train and validation sets
############################

# get sample indices not in train split
not_train = np.isin(np.arange(count_mtx.get_shape()[0]), 
                    np.concatenate((np.asarray(val_test_split['validation']),
                                    np.asarray(val_test_split['test']))),
                    invert=True)

# define heldout test indices
test_indices = np.asarray(val_test_split['test'])

testset = scDataset(count_mtx, meta_data, scaling_type='max', subset=test_indices)

batch_size = 32
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)

out_dim = testset.n_genes

############################
# define parameters
############################

# some model and training parameters
nsample_test = test_loader.dataset.__len__()

# dictionary with hyperparameters
dgd_hypers = {
    'latent': 20,
    'hidden': [100,100,100],
    'learning_rates': [1e-3,1e-2,1e-2],
    'lr_name': '322',
    'optim_beta': [0.5,0.7],
    'weight_decay': 1e-4,
    'dropout': 0.1,
    'gmm_type': 'diagonal',
    'n_gc': 18,
    'mp_scale': 1,
    'hardness': 1,
    'sd_mean': 0.01,
    'sd_sd': 1,
    'dirichlet_a': 1,
    'n_genes': out_dim
}

n_epochs = 500

model_name = 'dgd_latent20_components18'

############################
# initialize model
############################

save_dir = './results/'
# initialize GMM, representation and model and load parameters
mean_prior = softball(dgd_hypers['latent'],dgd_hypers['mp_scale'],dgd_hypers['hardness'])
gmm = GaussianMixture(Nmix=dgd_hypers['n_gc'], dim=dgd_hypers['latent'], type='diagonal', mean_prior=mean_prior, sd_init=(dgd_hypers['sd_mean'],dgd_hypers['sd_sd']), alpha=dgd_hypers['dirichlet_a'])
checkpoint = torch.load(save_dir+model_name+'/'+model_name+'_gmm.pt',map_location=torch.device('cpu'))
gmm.load_state_dict(checkpoint)
gmm = gmm.to(device)

model = DGD(latent=dgd_hypers['latent'], hidden=dgd_hypers['hidden'], out=out_dim, r_init=2, scaling_type='library')
checkpoint = torch.load(save_dir+model_name+'/'+model_name+'_decoder.pt',map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model = model.to(device)

############################
# train new rep
############################

f = open(save_dir+model_name+'/'+model_name+'_heldout_evaluation.txt', 'w')

resampling_epoch = [10,20]
resampling_type = 'mean'
resampling_samples = 1

f.write('resampling_epochs: '+str(resampling_epoch[0])+', '+str(resampling_epoch[1])+'\n')
f.write('resampling_type: '+resampling_type+'\n')
f.write('resampling_samples: '+str(resampling_samples)+'\n')
f.write(' '+'\n')

f.write('finding validation representations'+'\n')

def reshape_scaling_factor(x, o_dim):
    start_dim = len(x.shape)
    for _ in range(o_dim - start_dim):
        x = x.unsqueeze(1)
    return x

rep_init_values = gmm.sample_new_points(nsample_test, option=resampling_type, n_new=resampling_samples)
new_rep = RepresentationLayer(nrep=dgd_hypers['latent'],nsample=nsample_test,values=rep_init_values).to(device)
newrep_optimizer = torch.optim.Adam(new_rep.parameters(), lr=1e-2, weight_decay=0,betas=(0.5,0.7))
rep_init_values = None
torch.cuda.empty_cache()

model.eval()

for resample_epoch in range(resampling_epoch[0]):
    newrep_optimizer.zero_grad()
    for x,lib,i in test_loader:
        x = x.to(device)
        #lib = lib.unsqueeze(1)
        lib = lib.to(device)
        # x has shape (n,d)
        # z is chosen with i by intermediately viewing it as (n,m,c,l) and then shaping it to (n*m*c,l)
        z = gmm.reshape_targets(gmm.reshape_targets(new_rep.z,y_type='predicted')[i],y_type='reverse')
        y = model(z)
        # for calculating the loss correctly, x lib and y have to be brought to 4 dimensions
        ## x and y are viewed as (n,m,c,d) and lib is extended to (n,1,1,1)
        recon_loss_x = model.nb.loss(gmm.reshape_targets(x,y_type='true'), reshape_scaling_factor(lib, 4), gmm.reshape_targets(y,y_type='predicted')).sum()              
        gmm_error = - gmm(z).sum()
        loss = recon_loss_x.clone() + gmm_error.clone()
        loss.backward()
    newrep_optimizer.step()
x, lib, z, y = None, None, None, None
torch.cuda.empty_cache()

print('new samples trained')

f.write('   new samples trained'+'\n')

# if we choose new points via dataloader, we need a storage tensor for the new points
rep_new_values = torch.empty((nsample_test,dgd_hypers['latent']))
for x,lib,i in test_loader:
    x = x.to(device)
    lib = lib.to(device)
    # z is again chosen with i by intermediately viewing it as (n,m,c,l) and then shaping it to (n*m*c,l)
    z = gmm.reshape_targets(gmm.reshape_targets(new_rep.z,y_type='predicted')[i],y_type='reverse')
    y = model(z)
    # recon loss needs tensors in 4D again but is only summed over last dimension to have a loss for each component sample and point
    recon_loss_x = model.nb.loss(gmm.reshape_targets(x,y_type='true'), reshape_scaling_factor(lib, 4), gmm.reshape_targets(y,y_type='predicted')).sum(-1)             
    gmm_error = - gmm(z)
    # here, the recon loss has to be reshaped from (n,m,c) to (n*m*c) to match the gmm error
    loss = gmm.reshape_targets(recon_loss_x.clone(),y_type='reverse') + gmm_error.clone()
    # now we can easily get the best representations for i with just z and the loss
    rep_new_values[i] = gmm.choose_best_representations(z, loss).detach().cpu()

new_rep = RepresentationLayer(nrep=dgd_hypers['latent'],nsample=nsample_test,values=rep_new_values).to(device)
newrep_optimizer = torch.optim.Adam(new_rep.parameters(), lr=1e-2, weight_decay=0,betas=(0.5,0.7))
rep_new_values = None
x, lib, z, y = None, None, None, None
torch.cuda.empty_cache()

print('new samples selected')

f.write('   best samples chosen'+'\n')

newrep_loss = []
last_recon = 0
last_gmm = 0
for resample_epoch in range(resampling_epoch[1]):
    newrep_optimizer.zero_grad()
    newrep_loss.append(0)
    for x,lib,i in test_loader:
        x = x.to(device)
        #lib = lib.unsqueeze(1)
        lib = lib.to(device)
        #pos = find_matching_index(trainset.indices,i.numpy())
        z = new_rep(i)
        y = model(z)
        recon_loss_x = model.nb.loss(x, lib, y).sum()
        gmm_error = - gmm(z).sum()
        if resample_epoch == (resampling_epoch[1]-1):
            last_recon += recon_loss_x.clone().detach()/nsample_test
            last_gmm += gmm_error.clone().detach()/nsample_test
        loss = recon_loss_x.clone() + gmm_error.clone()
        loss.backward()

        newrep_loss[-1] += loss.item()/(nsample_test*out_dim)
    newrep_optimizer.step()
    f.write('      '+str(newrep_loss[-1])+'\n')
f.write('final reconstruction loss: '+str(last_recon)+'\n')
f.write('final gmm loss: '+str(last_gmm)+'\n')

print('done')

f.write('   best samples trained'+'\n')

torch.save(new_rep.state_dict(), save_dir+model_name+'/'+model_name+'_heldoutTestRepresentation.pt')