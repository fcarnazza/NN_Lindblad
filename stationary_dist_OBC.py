from ml.utils import update_params_from_cmdline, save_metrics_params, ensure_dir, ensure_empty_dir, infinite_dataset, xmap
from torch.utils.data import Dataset, DataLoader
from ml.pytorch_modules.vae import BVAE
from ml.pytorch_modules.dynamics_mlp import MLLP, MLP
from torch import optim
from train_data import *
from utils.settings import cfg
import os
#from tensorboardX import SummaryWriter #This module enable to monitor, live, the training later we will show
import numpy as np
import matplotlib.pyplot as plt
from physics.FSS_dyn.ED_b import *
import scipy.sparse.linalg as spa
import scipy.sparse as sps
from scipy import linalg as LA
import opt_einsum as oe
from get_Lindblad import  get_L_from_model

## Parameters
L = sub_n

dt = 0.01
T = 20 
nstep = int(T/dt)
Omega = 1.0
Delta = 0
C = 1
sites = [int(L/2),int(L/2)+1]


sigma_x = np.array( [ [ 0. , 1. ] , [ 1. , 0. ] ] )
N = np.array( [ [ 0. , 0. ] , [ 0. , 1. ] ] )
sigma_z = np.array( [ [ 1. , 0. ] , [ 0. , -1. ] ] )
sigma_y = np.array([[0, -1j], [1j, 0]])
Pj = np.array( [ [ 1. , 0. ] , [ 0. , 0. ] ] )
nf = np.array([[1, 0],[0, 0]]) ###  local fermions number nf = 1-sz ###
Id = np.eye(2).astype(np.float64)

pauli_dict = {
    'X': sigma_x,
    'Y': sigma_y,
    'Z': sigma_z,
    'I': Id
}
names=['I','X','Y','Z']
# Parameters of the header
default_parameters_test = {'device': 'cpu',
                           'alpha': [ sub_b],#sub_c, 0.3, 0.8, 1., 1.5, 2., 5., 8., sub_n.],
                           'potential': [sub_c],# sub_c, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0],
                            'MLP_par': {
                                'mlp_params': {
                                        'data_dim': 15,
                                        'layers':  [ ],
                              'nonlin': 'id',
                              'output_nonlin': 'id'
                                                                },
                          },
                      }


sigma_plus = np.array( [ [ 0. , 1. ] , [ 0. , 0. ] ] )
sigma_minus = np.array( [ [ 0. , 0. ] , [ 1. , 0. ] ] )
sigma_x = np.array([[0, 1], [1, 0]]) #### Rabi term####
sigma_z = np.array( [ [ 1. , 0. ] , [ 0. , -1. ] ] )
sigma_y = np.array([[0, 1j], [-1j, 0]])
Pj = np.array( [ [ 1. , 0. ] , [ 0. , 0. ] ] )
nf = np.array([[1, 0],[0, 0]]) ###  local fermions number nf = 1-sz ###
Id = np.eye(2)

#### Initial state ###
rho1 = np.array( [ [ 0.8 , 0 ],  [  0 , 0.2  ] ] ) #State of the sub-system
rho2 = np.array( [ [ 0.4 , 0 ],  [  0 , 0.6  ] ] )
rho = np.kron(rho1,rho2)

rho_loc=np.array([np.random.normal() for i in range(16)]).reshape(4,4)+1j*np.array([np.random.normal() for i in range(16)]).reshape(4,4)
rho_loc=rho_loc.dot(rho_loc.conjugate().transpose())

#red=rho_loc/rho_loc.trace()
red=rho_loc/rho_loc.trace()
 
#Thermal Hamiltonian for the sub-system with L - 2 sites
H_Rabi_th = H1f(sigma_x, L - 2)
H_Detuning_th = H1f(nf, L - 2)
H_int_th = H2srdef(nf, nf, L - 2)
H_th = ( Omega/2 )*H_Rabi_th + Delta*H_Detuning_th + H_int_th

conf = default_parameters_test
#conf_deep = default_parameters_test_deep
model = MLLP(**conf['MLP_par']).to(conf['device'])#we load the blueprint of the model
#model_trace = MLP(**conf_deep['MLP_par']).to(conf['device'])
#we load the saved parameters for 1 site impurity
#model.load_state_dict(torch.load('./trained_models/PBC/Thermal/eff__Cdef_%s_alpha_%s' %(C_def, alpha))) 

#Load the model for extended (2 sites) impurity
#model_trace.load_state_dict(torch.load('./trained_models/PBC/Thermal/eff__Cdef_%s_alpha_%s_trace_two_bodies_trace_loss_deep' %(C_def, alpha))) #we load the saved parameters


###### DYNAMICS ######


#der = 0
 
for pot in range(len(conf['potential'])):
 for inv_T in range(len(conf['alpha'])):
   paulis2, names_pauli2 = pauli_op(sites,L)
   tr_d=0
   eu_d=0
   C_def = conf['potential'][pot]
   alpha = conf['alpha'][inv_T] 


   F = {}
   for m in range(16):
         F [  names_pauli2[m] ]=np.kron(pauli_dict[ names_pauli2[m][0]],pauli_dict[names_pauli2[m][2]])
   H_Rabi = H1f(sigma_x, L)

   H_Detuning = H1f(nf, L)

   H_lr_ferm = H2fflr(nf, nf, L, alpha )

   H = ( Omega/2 )*H_Rabi + Delta*H_Detuning + C_def*H_lr_ferm


   print("HAMILTONIAN BUILT")


   model.load_state_dict(torch.load( './trained_models/OBC/Thermal/eff__alpha_%s_C_%s_n_spins_%s' %(alpha, C_def,L))) 
   bath = sps.eye(2**((L-2)/2))
   rho = np.kron(bath.toarray(),np.kron(rho_loc, bath.toarray()))
   e, U  = LA.eigh(H.toarray())

   rho_tilde = np.conj(U.T) @ rho @ U
   pa_ti = np.array([pa.toarray() for pa in paulis2])
   pauli2_tilde = np.conj(U.T) @ pa_ti @ U


   Lindblad = get_L_from_model( './trained_models/OBC/Thermal/eff__alpha_%s_C_%s_n_spins_%s' %(alpha, C_def,L))
   Lambdas, v_s = LA.eig(Lindblad)

   idx = np.real(Lambdas).argsort()[::-1]   
   Lambdas = Lambdas[idx]
   v_s = v_s[:,idx]
   gap =np.real( Lambdas)[1]
   v_stat = v_s.T[0]/v_s.T[0][0] 
   e_gap = np.exp(-1j*e*(5/gap ))
   rho_tilde = e_gap.reshape(e_gap.shape[0],1)*rho_tilde*np.conj(e_gap)
   trace = lambda k: ((oe.contract( 'ij,ji->', pauli2_tilde[k],rho_tilde ))/ rho_tilde.diagonal().sum()).real
   obs = xmap(trace,range(len( names_pauli2 ) ))

   v_in = np.array(obs)
   V_exact = {}
   for k in range(16):
        V_exact[names_pauli2[k]] = obs[k]
   dt = gap / 100
   for t in np.arange( 1, int(10/gap), 1/(100*gap)):
      print(t)
      ee = np.exp(-1j*e*t)
      rho_in = ee.reshape(ee.shape[0],1)*rho_tilde*np.conj(ee)
      rho_in = rho_in
      trace = lambda k: ((oe.contract( 'ij,ji->', pauli2_tilde[k],rho_in ))/ rho_in.diagonal().sum()).real
      obs = xmap(trace,range(len( names_pauli2 ) ))
      for k in range(16):
        V_exact[names_pauli2[k]] = np.append(V_exact[names_pauli2[k]], np.array(obs)[k] )
   v_stat_net = np.array([np.mean( V_exact[names_pauli2[k]]  ) for k in range(16) ])
   rho_diff = np.zeros(16,dtype = complex).reshape(4,4)
   for i in range(16):
         rho_diff += 0.25*(  v_stat_net[i] - v_stat[i] )*F[ names_pauli2[i]  ]
   print( v_stat_net  )
   print( v_stat  )
   l_diff, _ = LA.eigh(  rho_diff   )
   d_trace = np.sum( np.abs( l_diff  )  )
   print(d_trace )
   with open("stationary_diff_{sub_n}_OBC.txt", "a") as file_object:
    file_object.write("pot %f alpha %f  d_trace %f\n"%( C_def, alpha, d_trace )) #Matr[pot,inv_T] = tr_d/nstep

