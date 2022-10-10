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
   observables_dict_test={}




   for i in range(len(names_pauli2)):
        observables_dict_test[names_pauli2[i]]=((paulis2[i].dot(rho).diagonal().sum())/ 
                                           rho.diagonal().sum()).real


   v_in = torch.tensor([ observables_dict_test[key] for key in observables_dict_test.keys()  ][1:]).float()
#   temp1 = [v_in.numpy()]
   time = [ [0] ]
#   temp = [ [ observables_dict_test[key] for key in observables_dict_test.keys()  ][1:] ]
   tr_inter = 0
   eu_inter = 0 
   tr_extra = 0  
   eu_extra = 0  
   tr = 0  
   eu = 0  
   tr_t_minus_dt = 0
   for i in range( 0, nstep ):
     time +=  [ [ i * dt ] ]
     ee = np.exp(-1j*e*(dt*i))
     rho_in = ee.reshape(ee.shape[0],1)*rho_tilde*np.conj(ee)
     trace = lambda k: ((oe.contract( 'ij,ji->', pauli2_tilde[k],rho_in ))/ rho_in.diagonal().sum()).real
     obs = xmap(trace,range(len( names_pauli2 ) ))
     #Evolution with Exact diagonalization
     with torch.no_grad():
       data_in = torch.tensor(time[i])
       v_out = model(v_in.float())
       v_in = v_out
       #Check if the matrix obtained is actually positive
       r_recon = np.zeros(16,dtype=complex).reshape(4,4)
       r_exact = np.zeros(16,dtype=complex).reshape(4,4)
       for m in range(4): 
        for n in range(4):
         v_exact = np.array(obs) 
         rho_temp =  np.append(1,v_in.numpy()).reshape(4,4)
         r_recon += 0.25*rho_temp[m][n]*np.kron(pauli_dict[names[m]],pauli_dict[names[n]])
         r_exact += 0.25*v_exact.reshape(4,4)[m][n]*np.kron(pauli_dict[names[m]],pauli_dict[names[n]])
       r_recon /= r_recon.trace()
       r_exact /= r_exact.trace()
       Lambda,_ = LA.eigh(r_recon-r_exact)
       tr  = np.sum(np.abs(Lambda))
       eu  = np.sum(np.abs(v_in.numpy()-v_exact[1:] ) ) 
       with open("diff_traj_time_beta_{sub_b}_C_{sub_c}_{sub_n}_OBC_2.txt", "a") as file_object:
        file_object.write("time %f tr_d %f eu_d %f, d_over_dt_tr_d %f \n"%( i*dt,tr, eu, (tr-tr_t_minus_dt)/dt )) #Matr[pot,inv_T] = tr_d/nstep
       tr_t_minus_dt = tr
