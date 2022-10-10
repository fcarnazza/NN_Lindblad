import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .s_const import *
from math import sqrt
import numpy.linalg as LA
from .utils import *
nonlin_dict = {'relu': F.relu,
               'tanh': torch.tanh,
               'sigmoid': torch.sigmoid,
               'id': lambda x: x}

def get_arch_from_layer_list(input_dim, output_dim, layers):
    layers_module_list = nn.ModuleList([])
    layers = [input_dim]+layers+[output_dim]
    for i in range(len(layers)-1):
        layers_module_list.append(nn.Linear(layers[i], layers[i+1]))
    return layers_module_list



class MLPEncoder(nn.Module):
    def __init__(self, data_dim, latent_size, layers, nonlin, output_nonlin):
        super().__init__()
        self.layers = get_arch_from_layer_list(data_dim, layers[-1], layers[:-1])
        self.final_layer_mu = nn.Linear(layers[-1], latent_size)
        self.final_layer_log_sigma = nn.Linear(layers[-1], latent_size)
        self.nonlin = nonlin
        self.output_nonlin = output_nonlin

    def forward(self, x):
        for i, lin in enumerate(self.layers):
            x = lin(x)
            if not i == len(self.layers) - 1:
                x = nonlin_dict[self.nonlin](x)
        mus = nonlin_dict[self.output_nonlin](self.final_layer_mu(x))
        log_sigmas = nonlin_dict[self.output_nonlin](self.final_layer_log_sigma(x))
        return mus, log_sigmas

class MLPDecoder(nn.Module):
    def __init__(self, data_dim, latent_size, layers, nonlin, output_nonlin):
        super().__init__()
        self.layers = get_arch_from_layer_list(latent_size, data_dim, layers)
        self.nonlin = nonlin
        self.output_nonlin = output_nonlin

    def forward(self, z):
        for i, lin in enumerate(self.layers):
            z = lin(z)
            if not i == len(self.layers) - 1:
                z = nonlin_dict[self.nonlin](z)
        return nonlin_dict[self.output_nonlin](z)
        #return z

class MLP_drop(nn.Module):
    def __init__(self, data_dim, layers, nonlin, output_nonlin,p=0.5):
        super().__init__()
        self.layers = get_arch_from_layer_list(data_dim, data_dim, layers)
        self.nonlin = nonlin
        self.output_nonlin = output_nonlin
        self.dropout = nn.Dropout(0.000001)
    def forward(self, x):
        for i, lin in enumerate(self.layers):
            x = lin(x)
            x = self.dropout(x)
            #if not i == len(self.layers) - 1:
                #x = nonlin_dict[self.nonlin](x)
        #x = nonlin_dict[self.output_nonlin](x)

        return x



class MLP_simple(nn.Module):
    def __init__(self, data_dim, layers, nonlin, output_nonlin):
        super().__init__()
        self.layers = get_arch_from_layer_list(data_dim, data_dim, layers)
        self.nonlin = nonlin
        self.output_nonlin = output_nonlin

    def forward(self, x):
        for i, lin in enumerate(self.layers):
            x = lin(x)
            #if not i == len(self.layers) - 1:
                #x = nonlin_dict[self.nonlin](x)
        #x = nonlin_dict[self.output_nonlin](x)

        return x

class MLP_Hyper(nn.Module):
    def __init__(self, data_dim, layers, nonlin, output_nonlin):
        super().__init__()
        self.layers = get_arch_from_layer_list(1, data_dim**2, layers)
        self.nonlin = nonlin
        self.output_nonlin = output_nonlin

    def forward(self, x):
        for i, lin in enumerate(self.layers):
            x = lin(x)
            if not i == len(self.layers) - 1: 
                x = nonlin_dict[self.nonlin](x)
        x = nonlin_dict[self.output_nonlin](x)

        return x


class MLP_Hyper_old(nn.Module):
    def __init__(self, data_dim, layers, nonlin, output_nonlin):
        super().__init__()
        self.layers = get_arch_from_layer_list(1, data_dim**2, layers)
        self.nonlin = nonlin
        self.output_nonlin = output_nonlin

    def forward(self, x):
        for i, lin in enumerate(self.layers):
            x = lin(x)
            if not i == len(self.layers) - 1:
                x = nonlin_dict[self.nonlin](x)
        x = nonlin_dict[self.output_nonlin](x)
        return x


class LL(nn.Module):
    """ Custom Liouvillian layer to ensure positivity of the rho"""
    def __init__(self, data_dim, layers, nonlin, output_nonlin):
        super().__init__()
        self.layers = get_arch_from_layer_list(1, data_dim**2, layers)
        self.nonlin = nonlin
        self.output_nonlin = output_nonlin
        self.n = int(sqrt(data_dim+1))
        self.data_dim = data_dim
        # Dissipative parameters v = Re(v) + i Im(v) = x + i y
        v = torch.rand([self.data_dim, self.data_dim]).float()
        #self.v = nn.Parameter(v)
        self.v = nn.Parameter(v)
        # Hamiltonian parameters omega
        omega = torch.zeros([data_dim] )
        self.omega = nn.Parameter(omega).float()
        # initialize omega and v
        nn.init.kaiming_uniform_(self.v, a=1) 
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.v)
        bound = 1 / sqrt(fan_in)
        nn.init.uniform_(self.omega, -bound, bound)  # bias init
        #self.f,self.d = structure_const(self.n) #pauli_s_const()
        self.f,self.d = pauli_s_const()
        self.basis = get_basis(self.n)[1:]
    def forward(self, x):
        # Structure constant for SU(n) are defined
        # Structure constant are employed to massage the parameters omega and v into a completely positive dynamics
        idx = torch.triu_indices(self.data_dim,self.data_dim )
        idx_1 = torch.triu_indices(self.data_dim,self.data_dim, 1)
        c_re = torch.zeros(self.data_dim,self.data_dim)
        c_im = torch.zeros(self.data_dim,self.data_dim)
        c_re[idx[0],idx[1]] = self.v[idx[0],idx[1]]          
        c_re[idx[1],idx[0]] = self.v[idx[0],idx[1]]          
        c_im[idx_1[0],idx_1[1]] = self.v[idx_1[1],idx_1[0]]  
        c_im[idx_1[1],idx_1[0]] =-self.v[idx_1[1],idx_1[0]] 
        #c_re = torch.add(torch.einsum('ki,kj->ij',self.v_x,self.v_x), torch.einsum('ki,kj->ij',self.v_y,self.v_y)  )
        #c_im = torch.add(torch.einsum('ki,kj->ij',self.v_x,self.v_x), torch.einsum('ki,kj->ij',self.v_y,self.v_y)  )
        re_1 = -2.*torch.einsum('mjk,nik,ij->mn',self.f,self.f,c_re )
        re_2 = -2.*torch.einsum('mik,njk,ij->mn',self.f,self.f,c_re )
        im_1 = -2.*torch.einsum('mjk,nik,ij->mn',self.f,self.d,c_im )
        im_2 =  2.*torch.einsum('mik,njk,ij->mn',self.f,self.d,c_im )
        tr_id  = -8.*torch.einsum('imj,ij ->m',self.f,c_im )
        
        h_commutator_x =  8.* torch.einsum('ijk,k->ji', self.f, self.omega)
        d_super_x_re = torch.add(re_1,re_2 )
        d_super_x_im = torch.add(im_1,im_2 )
        d_super_x = torch.add(d_super_x_re,d_super_x_im )
        L = torch.zeros(self.data_dim+1,self.data_dim+1)
        L[1:,1:] = torch.add(h_commutator_x, 0.5*d_super_x)
        L[1:,0] = tr_id
        #exp_dt_L = torch.matrix_exp(0.01*L )
        exp_dt_L = torch.eye(self.data_dim+1) + 0.01*L  
        return torch.add(exp_dt_L[1:,0], x @ torch.transpose(exp_dt_L[1:,1:],0,1))  




class exp_LL(nn.Module):
    """ Custom Liouvillian layer to ensure positivity of the rho"""
    def __init__(self, data_dim, layers, nonlin, output_nonlin,dt=0.01):
        super().__init__()
        self.layers = get_arch_from_layer_list(1, data_dim**2, layers)
        self.nonlin = nonlin
        self.output_nonlin = output_nonlin
        self.n = int(sqrt(data_dim+1))
        self.data_dim = data_dim
        self.dt = dt
        self.f,self.d = pauli_s_const()
        # Dissipative parameters v = Re(v) + i Im(v) = x + i y
        v_re = torch.zeros([self.data_dim, self.data_dim],requires_grad=True).float()
        v_im = torch.zeros([self.data_dim, self.data_dim],requires_grad=True).float()
        self.v_x = nn.Parameter(v_re)  
        self.v_y = nn.Parameter(v_im) 
        # Hamiltonian parameters omega
        omega = torch.zeros([data_dim] )
        self.omega = nn.Parameter(omega).float()
        # initialize omega and v
        nn.init.kaiming_uniform_(self.v_x, a=1) 
        nn.init.kaiming_uniform_(self.v_y, a=1) 
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.v_x)
        bound = 1 / sqrt(fan_in)
        nn.init.uniform_(self.omega, -bound, bound)  # bias init
    def forward(self, x):
        # Structure constant for SU(n) are defined 
        #
        #We define the real and imaginary part od the Kossakowsky's matrix c.
        #       +
        # c = v   v =  ∑  x     x    + y   y    + i ( x   y  - y   x   )
        #              k    ki   kj     ki  kj         ki  kj   ki  kj
 
        c_re = torch.add(torch.einsum('ki,kj->ij',self.v_x,self.v_x), torch.einsum('ki,kj->ij',self.v_y,self.v_y)  )
        c_im = torch.add(torch.einsum('ki,kj->ij',self.v_x,self.v_y),-torch.einsum('ki,kj->ij',self.v_y,self.v_x)  ) 
        #
        # Structure constant are employed to massage the parameters omega and v into a completely positive dynamics.
        # Einsum not optimized in torch: https://optimized-einsum.readthedocs.io/en/stable/

        #re_1 = -2.*torch.einsum('mjk,nik,ij->mn',self.f,self.f,c_re )
        #re_2 = -2.*torch.einsum('mik,njk,ij->mn',self.f,self.f,c_re )
        #im_1 = -2.*torch.einsum('mjk,nik,ij->mn',self.f,self.d,c_im )
        #im_2 =  2.*torch.einsum('mik,njk,ij->mn',self.f,self.d,c_im )
        #tr_id  = -8.*torch.einsum('imj,ij ->m',self.f,c_im )
        #h_commutator_x =  8.* torch.einsum('ijk,k->ji', self.f, self.omega)
        re_1 = -4.*torch.einsum('mjk,nik,ij->mn',self.f,self.f,c_re )
        re_2 = -4.*torch.einsum('mik,njk,ij->mn',self.f,self.f,c_re )
        im_1 = -4.*torch.einsum('mjk,nik,ij->mn',self.f,self.d,c_im )
        im_2 =  4.*torch.einsum('mik,njk,ij->mn',self.f,self.d,c_im )
        tr_id  = -4.*torch.einsum('imj,ij ->m',self.f,c_im )
        h_commutator_x =  4.* torch.einsum('ijk,k->ji', self.f, self.omega)
        d_super_x_re = torch.add(re_1,re_2 )
        d_super_x_im = torch.add(im_1,im_2 )
        d_super_x = torch.add(d_super_x_re,d_super_x_im )
        L = torch.zeros(self.data_dim+1,self.data_dim+1)
        L[1:,1:] = torch.add(h_commutator_x, 0.5*d_super_x)
        L[1:,0] = tr_id
        exp_dt_L = torch.matrix_exp(self.dt*L )
        return torch.add(exp_dt_L[1:,0], x @ torch.transpose(exp_dt_L[1:,1:],0,1))  



class tri_LL(nn.Module):
    """ Custom Liouvillian layer to ensure positivity of the rho"""
    def __init__(self, data_dim, layers, nonlin, output_nonlin):
        super().__init__()
        self.layers = get_arch_from_layer_list(1, data_dim**2, layers)
        self.nonlin = nonlin
        self.output_nonlin = output_nonlin
        self.n = int(sqrt(data_dim+1))
        self.data_dim = data_dim
        # Dissipative parameters v = Re(v) + i Im(v) = x + i y
        self.tri_ind = torch.triu_indices(self.data_dim,self.data_dim )
        self.tri_ind_1 = torch.triu_indices(self.data_dim,self.data_dim, 1)
        r_re = torch.rand([int(self.data_dim*(self.data_dim+1)*0.5)],requires_grad=True).float()
        r_im = torch.rand([int(self.data_dim*(self.data_dim-1)*0.5)],requires_grad=True).float()
        n = torch.rand([1],requires_grad=True).float()
        self.N = nn.Parameter(n)       
        self.r_x = nn.Parameter(r_re)  
        self.r_y = nn.Parameter(r_im) 
        # Hamiltonian parameters omega
        omega = torch.rand([data_dim] )
        self.omega = nn.Parameter(omega).float()
        # initialize omega and v
        #nn.init.sparse_(self.v_x, sparsity=0.4,std=sqrt(5)) 
        #nn.init.sparse_(self.v_y, sparsity=0.4,std=sqrt(5)) 
        #nn.init.kaiming_uniform_(self.v_y, a=1) 
        #nn.init.kaiming_uniform_(self.v_x, a=1) 
        #fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.r_x)
        #bound = 1 / sqrt(fan_in)
        #nn.init.uniform_(self.omega, -bound, bound)  # bias init
    def forward(self, x):
        # Structure constant for SU(n) are defined
        f,d = structure_const(self.n)
        # Structure constant are employed to massage the parameters omega and v into a completely positive dynamics
        if False: #x.requires_grad():
                c_re = c_im = torch.zeros(self.data_dim,self.data_dim) 
                c_im[self.tri_ind[0], self.tri_ind[1]] = self.r_x
                c_re[self.tri_ind_1[0], self.tri_ind_1[1]] = self.r_y
                c_re[self.tri_ind_1[1], self.tri_ind_1[0] ] = c_re[self.tri_ind_1[0], self.tri_ind_1[1]  ] 
                c_im[self.tri_ind_1[1], self.tri_ind_1[0] ] = -c_im[self.tri_ind_1[0], self.tri_ind_1[1]  ] 
               # c_re = 0.5*(torch.transpose(self.v_x,0,1)+self.v_x)# torch.add(torch.einsum('ki,kj->ij',self.v_x,self.v_x), torch.einsum('ki,kj->ij',self.v_y,self.v_y)  )
               # c_im = 0.5*(torch.transpose(self.v_y,0,1)-self.v_y)# torch.add(torch.einsum('ki,kj->ij',self.v_x,self.v_y),-torch.einsum('ki,kj->ij',self.v_y,self.v_x)  ) 
                re_1 = -2.*torch.einsum('mjk,nik,ij,n->m',f,f,c_re,x )
                re_2 = -2.*torch.einsum('mik,njk,ij,n->m',f,f,c_re,x )
                im_1 = -2.*torch.einsum('mjk,nik,ij,n->m',f,d,c_im,x )
                im_2 =  2.*torch.einsum('mik,njk,ij,n->m',f,d,c_im,x )
                tr_id  = -8.*torch.einsum('imj,ij ->m',f,c_im )
                h_commutator_x =  8.* torch.einsum('kij,k,j->i', f, self.omega, x)
        else:
                v_x = v_y = torch.zeros(self.data_dim,self.data_dim) 
                v_x[self.tri_ind[0], self.tri_ind[1]] = self.r_x
                v_y[self.tri_ind_1[0], self.tri_ind_1[1]] = self.r_y
                #c_re[self.tri_ind_1[1], self.tri_ind_1[0] ] = c_re[self.tri_ind_1[0], self.tri_ind_1[1]  ] 
                #c_im[self.tri_ind_1[1], self.tri_ind_1[0] ] = -c_im[self.tri_ind_1[0], self.tri_ind_1[1]  ] 
  
                c_re = torch.add(torch.einsum('ki,kj->ij',v_x,v_x), torch.einsum('ki,kj->ij',v_y,v_y)  )
                c_re = c_re/torch.trace(c_re)
                c_im = torch.add(torch.einsum('ki,kj->ij',v_x,v_y),-torch.einsum('ki,kj->ij',v_y,v_x)  ) 
                c_re = self.N*c_re
                re_1 = -2.*torch.einsum('mjk,nik,ij,bn->bm',f,f,c_re,x )
                re_2 = -2.*torch.einsum('mik,njk,ij,bn->bm',f,f,c_re,x )
                im_1 = -2.*torch.einsum('mjk,nik,ij,bn->bm',f,d,c_im,x )
                im_2 =  2.*torch.einsum('mik,njk,ij,bn->bm',f,d,c_im,x )
                tr_id  = -8.*torch.einsum('imj,ij ->m',f,c_im )
                h_commutator_x = 8.* torch.einsum('kij,k,bj->bi', f, self.omega, x)
        d_super_x_re = torch.add(re_1,re_2 )
        d_super_x_im = torch.add(im_1,im_2 )
        d_super_x = torch.add(d_super_x_re,d_super_x_im )
        S_x = h_commutator_x + d_super_x 
        T_x = torch.add(x,tr_id )
        L_x = torch.add(S_x,T_x )
        return torch.add(x,0.01*L_x )




class gmLL(nn.Module):
    """ Custom Liouvillian layer to ensure positivity of the rho"""
    def __init__(self, data_dim, layers, nonlin, output_nonlin):
        super().__init__()
        self.layers = get_arch_from_layer_list(1, data_dim**2, layers)
        self.nonlin = nonlin
        self.output_nonlin = output_nonlin
        self.n = int(sqrt(data_dim+1))
        self.data_dim = data_dim
        self.basis = torch.from_numpy(get_basis(self.data_dim )[:int(self.data_dim**2-1)])
        # Bloch vector of Kossakowski's matrix and its normalization tr initialized
        #c0 = np.random.randn(data_dim,data_dim) + 1j*np.random.randn(data_dim,data_dim)
        #c0 = c0 @ np.conj(c0.T)
        #c0 = torch.from_numpy(c0/c0.trace())
        #bloch = 0.5*torch.real(torch.from_numpy(np.einsum('ijk,kj->i',self.basis.numpy() ,c0 ))).float()
        self.bloch = torch.rand([int(self.data_dim**2-1) ],requires_grad=True ).float()
        self.tr = torch.rand([1],requires_grad=True).float()
        self.Tr = nn.Parameter(self.tr)
        self.Bloch = nn.Parameter(self.bloch)
        self.basis_re = torch.real(self.basis).float()
        self.basis_im = torch.imag(self.basis).float()
        # Hamiltonian parameters omega
        omega = torch.rand([data_dim] )
        self.omega = nn.Parameter(omega).float()
    def forward(self, x):
        # Structure constant for SU(n) are defined
        f,d = structure_const(self.n)
        # Structure constant are employed to massage the parameters omega and v into a completely positive dynamics
        if False: #x.requires_grad():
                c_re =  self.Tr**2/(self.data_dim )*(torch.eye(self.data_dim)  + torch.einsum('kij,k->ij',self.basis_re ,self.Bloch)) 
                c_im =  self.Tr**2/(self.data_dim )*torch.einsum('kij,k->ij',self.basis_im ,self.Bloch)
                re_1 = -2.*torch.einsum('mjk,nik,ij,n->m',f,f,c_re,x )
                re_2 = -2.*torch.einsum('mik,njk,ij,n->m',f,f,c_re,x )
                im_1 = -2.*torch.einsum('mjk,nik,ij,n->m',f,d,c_im,x )
                im_2 =  2.*torch.einsum('mik,njk,ij,n->m',f,d,c_im,x )
                tr_id  = -8.*torch.einsum('imj,ij ->m',f,c_im )
                h_commutator_x =  8.* torch.einsum('kij,k,j->i', f, self.omega, x)
        else:
                c_re =  self.Tr**2/(self.data_dim )*(torch.eye(self.data_dim)  + torch.einsum('kij,k->ij',self.basis_re ,self.Bloch)) 
                c_im =  self.Tr**2/(self.data_dim )*torch.einsum('kij,k->ij',self.basis_im ,self.Bloch)
                #c_re =  self.Tr**2*(torch.eye(self.data_dim )/self.data_dim + torch.einsum('kij,k->ij',self.basis_re ,self.Bloch)) 
                #c_im =  self.Tr**2*torch.einsum('kij,k->ij',self.basis_im ,self.Bloch)
                re_1 = -2.*torch.einsum('mjk,nik,ij,bn->bm',f,f,c_re,x )
                re_2 = -2.*torch.einsum('mik,njk,ij,bn->bm',f,f,c_re,x )
                im_1 = -2.*torch.einsum('mjk,nik,ij,bn->bm',f,d,c_im,x )
                im_2 =  2.*torch.einsum('mik,njk,ij,bn->bm',f,d,c_im,x )
                tr_id  = -8.*torch.einsum('imj,ij ->m',f,c_im )
                h_commutator_x = 8.* torch.einsum('kij,k,bj->bi', f, self.omega, x)
        d_super_x_re = torch.add(re_1,re_2 )
        d_super_x_im = torch.add(im_1,im_2 )
        d_super_x = torch.add(d_super_x_re,d_super_x_im )
        S_x = h_commutator_x + d_super_x 
        T_x = torch.add(x,tr_id )
        L_x = torch.add(S_x,T_x )
        return torch.add(x,0.01*L_x )



