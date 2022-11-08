import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from DGD.Prior import GaussianMixture, softball
from DGD.Representation import RepresentationLayer
from src.model import FmnistDGD
from src.functions import FashionMNIST_Dataset, SingleGaussian
from src.functions import set_random_seed, train

############################
# get the data and prepare loaders
############################

trainset = FashionMNIST_Dataset(train=True)
testset = FashionMNIST_Dataset(train=False)
classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
           'Sandal','Shirt','Sneaker','Bag','Ankle boot']

batch_size = 512
seed = 0
set_random_seed(seed)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=True, num_workers=0)

############################
# define parameters
############################

nepochs = 500 # default

save_dir = './results/paper_experiments/'

dgd_hypers = {
    'capacity': 32,
    'skip': True,
    'pixel': True,
    'pixel_specs': [(5),(5)],
    'latent': 20, # default
    'hidden': True,
    'hidden_dim': 100,
    'learning_rates': [[1e-3,1e-2,1e-1]], # default
    'lr_schedule': [0], # default
    'lr_name': '321',
    'optim_beta': [0.5,0.7], # default
    'weight_decay': 1e-4, # default
    'dropout': 0.1, # default
    'gmm_type': 'diagonal',
    'n_gc': 20,
    'activ': 'silu',
    'block_type': 'simple',
    'rep_init': 'zero',
    'mean_prior': 'Softball',
    'mean_a': 1,
    'mean_b': 1,
    'sd_mean': 1,
    'sd_sd': 1,
    'dirichlet_a': 1,
}

model_name = 'dgd_fmnist'

############################
# initialize model
############################

decoder = FmnistDGD(latent=dgd_hypers['latent'], hidden=dgd_hypers['hidden'], hidden_dim=dgd_hypers['hidden_dim'], 
    capacity=dgd_hypers['capacity'], activ=dgd_hypers['activ'], pixel=dgd_hypers['pixel'], skip = dgd_hypers['skip'], dropout=dgd_hypers['dropout'], 
    pixel_layers=dgd_hypers['pixel_specs'][0],pixel_kernels=dgd_hypers['pixel_specs'][1])

representation = RepresentationLayer(dgd_hypers['latent'],trainloader.dataset.__len__(),values=torch.zeros(size=(trainloader.dataset.__len__(),dgd_hypers['latent'])))
test_representation = RepresentationLayer(dgd_hypers['latent'],testloader.dataset.__len__(),values=torch.zeros(size=(testloader.dataset.__len__(),dgd_hypers['latent'])))

if dgd_hypers['n_gc'] is not None:
    if dgd_hypers['n_gc'] == 0:
        gmm = SingleGaussian(dim=dgd_hypers['latent'])
    else:
        mean_prior = softball(dgd_hypers['latent'],dgd_hypers['mean_b'],dgd_hypers['mean_a'])
        gmm = GaussianMixture(Nmix=dgd_hypers['n_gc'], dim=dgd_hypers['latent'], type=dgd_hypers['gmm_type'], mean_prior=mean_prior, sd_init=(dgd_hypers['sd_mean'],dgd_hypers['sd_sd']), alpha=dgd_hypers['dirichlet_a'])
else:
    gmm = None

rep_optimizer = torch.optim.Adam(representation.parameters(), lr=dgd_hypers['learning_rates'][0][2], weight_decay=dgd_hypers['weight_decay'],betas=(dgd_hypers['optim_beta'][0],dgd_hypers['optim_beta'][1]))
testrep_optimizer = torch.optim.Adam(test_representation.parameters(), lr=dgd_hypers['learning_rates'][0][2], weight_decay=dgd_hypers['weight_decay'],betas=(dgd_hypers['optim_beta'][0],dgd_hypers['optim_beta'][1]))

############################
# train
############################

train(model=decoder, gmm=gmm, rep=representation, test_rep=test_representation,
    data=trainloader,testdata=testloader,latent=dgd_hypers['latent'],
    rep_optimizer=rep_optimizer,testrep_optimizer=testrep_optimizer,
    device=device,nepochs=nepochs,export_dir=save_dir,export_name=model_name)
