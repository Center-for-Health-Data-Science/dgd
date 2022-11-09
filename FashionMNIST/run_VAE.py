import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.functions import FashionMNIST_Dataset
from src.functions import set_random_seed, train
from src.functions_classes_VI import VAE

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
    'latent': 100, # default
    'hidden': True,
    'hidden_dim': 100,
    'learning_rates': [[1e-3]], # default
    'lr_schedule': [0], # default
    'lr_name': '3',
    'optim_beta': [0.9,0.99], # default
    'weight_decay': 1e-4, # default
    'dropout': 0.1, # default
    'n_gc': 0,
    'activ': 'silu',
    'block_type': 'simple'
}

model_name = 'vae_fmnist'

############################
# initialize model
############################

model = VAE(latent=dgd_hypers['latent'], hidden=dgd_hypers['hidden'], hidden_dim=dgd_hypers['hidden_dim'], 
    capacity=dgd_hypers['capacity'], activ=dgd_hypers['activ'], pixel=dgd_hypers['pixel'], skip = dgd_hypers['skip'], 
    pixel_layers=dgd_hypers['pixel_specs'][0],pixel_kernels=dgd_hypers['pixel_specs'][1])

############################
# train
############################

train(model=model, gmm=None, rep=None, test_rep=None,
    data=trainloader,testdata=testloader,latent=dgd_hypers['latent'],
    export_dir=save_dir,export_name=model_name,
    model_type='vae')
