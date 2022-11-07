import os
import torch
import numpy as np
from torch.utils.data import Dataset
from DGD.Representation import RepresentationLayer

class scDataset(Dataset):
    """
    """
    def __init__(self, sparse_mtrx, meta_data, scaling_type='mean',gene_selection=None, subset=None):
        """
        Args:
        """
        if gene_selection is not None:
            sparse_mtrx = sparse_mtrx.tocsc()[:,gene_selection].tocoo()
        if subset is not None:
            sparse_mtrx = sparse_mtrx.tocsr()[subset]
        
        self.data = torch.Tensor(sparse_mtrx.todense())
        self.scaling_type = scaling_type
        self.meta = meta_data
        #self.gene_subset = gene_selection
        self.n_genes = self.data.shape[1]

        if self.scaling_type == 'mean':
            self.library = torch.mean(self.data, dim=-1).unsqueeze(1)
        elif self.scaling_type == 'max':
            self.library = torch.max(self.data, dim=-1).values.unsqueeze(1)

    def __len__(self):
        return(self.data.shape[0])

    def __getitem__(self, idx=None):
        if idx is not None:
          expression = self.data[idx]
        else:
          expression = self.data
          idx = torch.arange(self.data.shape[0])
        lib = self.library[idx]
        return expression, lib, idx
    
    def get_labels(self, idx=None):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx is None:
            idx = np.arange(self.__len__())
        label_ids = np.argmax(np.expand_dims(np.asarray(self.meta['stop']),0)>=np.expand_dims(idx,1),axis=1)
        return np.asarray(np.array(self.meta['label'])[label_ids])
    
    def get_labels_numerical(self, idx=None):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx is None:
            idx = np.arange(self.__len__())
        label_ids = np.argmax(np.expand_dims(np.asarray(self.meta['stop']),0)>=np.expand_dims(idx,1),axis=1)
        return label_ids

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def dgd_train(model, gmm, latent,
            train_loader, validation_loader,
            nsample, nsample_test, n_epochs, export_dir, export_name, device,
            pretrain=0, lr_schedule_epochs=None,lr_schedule=[1e-3,1e-2,1e-1],optim_betas=[0.5,0.7],wd=1e-4,
            start_epoch=0,early_stopping=True):

    if export_name is not None:
        if not os.path.exists(export_dir+export_name):
            os.makedirs(export_dir+export_name)
    
    out_dim = train_loader.dataset.n_genes

    model=model.to(device)
    gmm=gmm.to(device)

    if lr_schedule_epochs is None:
        lr = lr_schedule[0]
        lr_rep = lr_schedule[1]
        lr_gmm = lr_schedule[2]
    else:
        lr = lr_schedule[0][0]
        lr_rep = lr_schedule[0][1]
        lr_gmm = lr_schedule[0][2]

    rep = RepresentationLayer(nrep=latent,nsample=nsample,values=torch.zeros(size=(nsample,latent))).to(device)
    test_rep = RepresentationLayer(nrep=latent,nsample=nsample_test,values=torch.zeros(size=(nsample_test,latent))).to(device)
    rep_optimizer = torch.optim.Adam(rep.parameters(), lr=lr_rep, weight_decay=wd,betas=(optim_betas[0],optim_betas[1]))
    testrep_optimizer = torch.optim.Adam(test_rep.parameters(), lr=lr_rep, weight_decay=wd,betas=(optim_betas[0],optim_betas[1]))

    if gmm is not None:
        gmm_optimizer = torch.optim.Adam(gmm.parameters(), lr=lr_gmm, weight_decay=wd,betas=(optim_betas[0],optim_betas[1]))
    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd,betas=(optim_betas[0],optim_betas[1]))

    # start training
    train_loss = []
    test_loss = []

    best_test_loss = None
    best_test_epoch = 0

    for epoch in range(start_epoch,n_epochs):

        train_loss.append(0)
        test_loss.append(0)

        # train
        rep_optimizer.zero_grad()

        for x,lib,i in train_loader:
            mbatch_size = x.shape[0]
            x = x.to(device)
            lib = lib.to(device)
            train_gmm = False
            if gmm is not None:
                if pretrain < epoch:
                    gmm_optimizer.zero_grad()
                    train_gmm = True
            model_optimizer.zero_grad()

            z = rep(i)
            y = model(z)

            recon_loss_x = model.nb.loss(x, lib, y).sum()
            if train_gmm:
                gmm_error = - gmm(z).sum()
                loss = recon_loss_x.clone() + gmm_error.clone()
            else:
                gmm_error = 0
                loss = recon_loss_x.clone()
            loss.backward()
            
            if train_gmm:
                gmm_optimizer.step()
            model_optimizer.step()

            train_loss[-1] += loss.item()/(mbatch_size*out_dim)
        rep_optimizer.step()

        # test
        testrep_optimizer.zero_grad()
        for x,lib,i in validation_loader:
            mbatch_size = x.shape[0]
            x = x.to(device)
            lib = lib.to(device)
            z = test_rep(i)
            y = model(z)
            recon_loss_x = model.nb.loss(x, lib, y).sum()
            if train_gmm:
                gmm_error = - gmm(z).sum()
                loss = recon_loss_x.clone() + gmm_error.clone()
            else:
                gmm_error = 0
                loss = recon_loss_x.clone()
            loss.backward()

            test_loss[-1] += loss.item()/(mbatch_size*out_dim)
        testrep_optimizer.step()

        save_here = False
        if best_test_loss is not None:
            if best_test_loss < test_loss[-1]:
                save_here = True
                best_test_loss = test_loss[-1]
                best_test_epoch = epoch
        else:
            best_test_loss = test_loss[-1]
        
        if early_stopping:
            if save_here:
                print("model saved at epoch "+str(epoch))
                torch.save(model.state_dict(), export_dir+export_name+'/'+export_name+'_decoder.pt')
                torch.save(rep.state_dict(), export_dir+export_name+'/'+export_name+'_representation.pt')
                torch.save(test_rep.state_dict(), export_dir+export_name+'/'+export_name+'_testRepresentation.pt')
                torch.save(gmm.state_dict(), export_dir+export_name+'/'+export_name+'_gmm.pt')