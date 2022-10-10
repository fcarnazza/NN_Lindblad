import torch
from ml.pytorch_modules.architectures import *
import numpy as np
from scipy import linalg as LA
import opt_einsum as oe
from .utils import *
class MLP(nn.Module):
    def __init__(self, mlp_params, rec_loss_fn=torch.nn.MSELoss()):
        super().__init__()
        self.rec_loss_fn = rec_loss_fn
        self.MLP = MLP_simple(**mlp_params)
        #self.MLP = LL(**mlp_params)
        
    def forward(self, x):
        return self.MLP(x)

    def trace_loss(self,x, recon_x):
        unity = [[1.,0.],[0.,1.]] 
        c_unity = [[-1.,0.],[0.,1.]] 
        zero = torch.zeros(2,2)
        paulis = torch.tensor([[[1.,0.,0.,0.],
                                [0.,1.,0.,0.],
                                [0.,0.,1.,0.],
                                [0.,0.,0.,1.]],
                               [[0.,0.,1.,0.],
                                [0.,0.,0.,1.],
                                [1.,0.,0.,0.],
                                [0.,1.,0.,0.]],
                               [[0.,0.,0.,1.],
                                [0.,0.,-1.,0.],
                                [0.,-1.,0.,0.],
                                [1.,0.,0.,0.]],
                               [[1.,0.,0.,0.],
                                [0.,1.,0.,0.],
                                [0.,0.,-1.,0.],
                                [0.,0.,0.,-1.]] ])
        batch_size = len(x)
        p = recon_x-x
        #x = x.reshape(batch_size,4,4)
        #recon_x = recon_x.reshape(batch_size,4,4)
        s_mn = torch.zeros(4,4,16,16)
        for m in range(4):
            for n in range(4):
                s_mn[m,n] = kronecker_product(paulis[m],paulis[n])
        s_mn = s_mn.reshape(16,16,16)[1:]
        pp = oe.contract('bx,xkl->bkl',p,s_mn )
        #sigma,U = torch.symeig(pp , eigenvectors = True )
        #sigma   = (torch.abs(sigma)+sigma)/2.
        #sigma  /= sigma.sum()
        #rt = oe.contract('bik,bk,bkl->bil',U,sigma,torch.transpose(U,1,2))
        e,_ = torch.symeig(pp , eigenvectors = True )
        loss= torch.sum(torch.abs(e),1)
        return torch.mean(loss)

    def corr_trace_loss(self,x, recon_x):
        p = x-recon_x
        se_loss = torch.sum((p )**2,1) 
        batch_size = len(x)
        p=p.reshape(batch_size,4,4)
        corr = torch.einsum('bij,bkl,bkj,bil->b',p,p,p,p )
        loss = (3* se_loss + corr)
        return torch.mean(loss)

    def loss(self, x, recon_x):
        rec = self.rec_loss_fn(recon_x, x)
        return rec

class hyper_mlp(nn.Module):
    def __init__(self, mlp_params, rec_loss_fn=torch.nn.MSELoss()):
        super().__init__()
        self.rec_loss_fn = rec_loss_fn
        self.MLP = MLP_Hyper(**mlp_params)
        self.mlp_params = mlp_params
    def forward(self, t, x):
        M = self.MLP(t)
        size = M.shape
        if len(size)>1:
        	M = M.reshape([size[0], self.mlp_params['data_dim'], self.mlp_params['data_dim']  ])
        else:
        	M = M.reshape([self.mlp_params['data_dim'],self.mlp_params['data_dim'] ])
        x = x.unsqueeze(-1)
        r =  M @ x
        #print(r.shape)
        return r.squeeze(-1), M


    def loss(self, x, recon_x):
        rec = self.rec_loss_fn(recon_x, x)
        return rec



class hyper_mlp_old(nn.Module):
    
    def __init__(self, mlp_params, rec_loss_fn=torch.nn.MSELoss()):
        super().__init__()
        self.rec_loss_fn = rec_loss_fn
        self.MLP = MLP_Hyper_old(**mlp_params)

    def forward(self, t, x):
        M = self.MLP(t)
        size = M.shape
        if len(size)>1:
            M = M.reshape([size[0], 3, 3])
        else:
            M = M.reshape([3,3])
        x = x.unsqueeze(-1)
        r =  M @ x
        #print(r.shape)
        return r.squeeze(-1), M

    def loss(self, x, recon_x):
        rec = self.rec_loss_fn(recon_x, x)
        return rec

class MLLP(nn.Module):
    def __init__(self, mlp_params, rec_loss_fn=torch.nn.MSELoss()):
        super().__init__()
        self.rec_loss_fn = rec_loss_fn
        self.MLP = exp_LL(**mlp_params)
        
    def forward(self, x):
        return self.MLP(x)
    def trace_loss(self,x, recon_x):
        unity = [[1.,0.],[0.,1.]] 
        c_unity = [[-1.,0.],[0.,1.]] 
        zero = torch.zeros(2,2)
        paulis = torch.tensor([[[1.,0.,0.,0.],
                                [0.,1.,0.,0.],
                                [0.,0.,1.,0.],
                                [0.,0.,0.,1.]],
                               [[0.,0.,1.,0.],
                                [0.,0.,0.,1.],
                                [1.,0.,0.,0.],
                                [0.,1.,0.,0.]],
                               [[0.,0.,0.,1.],
                                [0.,0.,-1.,0.],
                                [0.,-1.,0.,0.],
                                [1.,0.,0.,0.]],
                               [[1.,0.,0.,0.],
                                [0.,1.,0.,0.],
                                [0.,0.,-1.,0.],
                                [0.,0.,0.,-1.]] ])
        batch_size = len(x)
        p = recon_x-x
        #x = x.reshape(batch_size,4,4)
        #recon_x = recon_x.reshape(batch_size,4,4)
        s_mn = torch.zeros(4,4,16,16)
        for m in range(4):
            for n in range(4):
                s_mn[m,n] = kronecker_product(paulis[m],paulis[n])
        s_mn = s_mn.reshape(16,16,16)[1:]
        pp = oe.contract('bx,xkl->bkl',p,s_mn )
        #sigma,U = torch.symeig(pp , eigenvectors = True )
        #sigma   = (torch.abs(sigma)+sigma)/2.
        #sigma  /= sigma.sum()
        #rt = oe.contract('bik,bk,bkl->bil',U,sigma,torch.transpose(U,1,2))
        e,_ = torch.symeig(pp , eigenvectors = True )
        loss= torch.sum(torch.abs(e),1)
        return torch.mean(loss)


    def loss(self, x, recon_x):
        rec = self.rec_loss_fn(recon_x, x)
        return rec











class tri_LLP(nn.Module):
    def __init__(self, mlp_params, rec_loss_fn=torch.nn.MSELoss()):
        super().__init__()
        self.rec_loss_fn = rec_loss_fn
        self.MLP = tri_LL(**mlp_params)
        
    def forward(self, x):
        return self.MLP(x)


    def loss(self, x, recon_x):
        rec = self.rec_loss_fn(recon_x, x)
        return rec








