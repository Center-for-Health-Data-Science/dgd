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
from src.functions import set_random_seed, dgd_train


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

# define train and validation indices
train_indices = np.arange(count_mtx.get_shape()[0])[not_train].tolist()
val_indices = np.asarray(val_test_split['validation'])

trainset = scDataset(count_mtx, meta_data, scaling_type='max', subset=train_indices)
validationset = scDataset(count_mtx, meta_data, scaling_type='max', subset=val_indices)

batch_size = 256
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)
validation_loader = torch.utils.data.DataLoader(validationset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)

out_dim = trainset.n_genes

############################
# define parameters
############################

# some model and training parameters
nsample = train_loader.dataset.__len__()
nsample_test = validation_loader.dataset.__len__()

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
    'n_genes': trainset.n_genes
}

n_epochs = 500

model_name = 'dgd_latent20_components18'

############################
# initialize model
############################

# initialize GMM, representation, model and optimizers
mean_prior = softball(dgd_hypers['latent'],dgd_hypers['mp_scale'],dgd_hypers['hardness'])
gmm = GaussianMixture(Nmix=dgd_hypers['n_gc'], dim=dgd_hypers['latent'], 
    type='diagonal', mean_prior=mean_prior, sd_init=(dgd_hypers['sd_mean'],dgd_hypers['sd_sd']), 
    alpha=dgd_hypers['dirichlet_a'])

model = DGD(latent=dgd_hypers['latent'], hidden=dgd_hypers['hidden'], 
    out=out_dim, r_init=2, scaling_type='library')

############################
# train
############################

dgd_train(model, gmm, dgd_hypers['latent'], train_loader, validation_loader,
        nsample, nsample_test, n_epochs,
        export_dir='./results/', export_name=model_name, device=device,
        lr_schedule_epochs=dgd_hypers['lr_schedule'],lr_schedule=dgd_hypers['learning_rates'],
        early_stopping=True)