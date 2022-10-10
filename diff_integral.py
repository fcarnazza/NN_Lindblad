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

from scipy.spatial import distance


## Parameters
L = sub_n

dt = 0.01
T = 10 
nstep = int(T/dt)
Omega = 1.0
Delta = 0
C = 1
sites = [1,2]


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
                           'beta': [sub_b],#0.1, 0.3, 0.8, 1., 1.5, 2., 5., 8., 10.],
                           'potential': [sub_c],# 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0],
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
#State of the bath

#temp1_trace = [v_in.numpy()[1:]]
conf = default_parameters_test
#conf_deep = default_parameters_test_deep
model = MLLP(**conf['MLP_par']).to(conf['device'])#we load the blueprint of the model

###### DYNAMICS ######


#der = 0
Matr = np.zeros(( len(conf['potential'])  , len(conf['beta'])  ) )
 
for pot in range(len(conf['potential'])):
 for inv_T in range(len(conf['beta'])):
   paulis2, names_pauli2 = pauli_op(sites,L)
   tr_d=0
   eu_d=0
   C_def = conf['potential'][pot]
   beta = conf['beta'][inv_T] 

   H_Rabi = H1f(sigma_x, L)
   H_Detuning = H1f(nf, L)
   H_int = H3srdef(nf, nf, L)
   H_def = H3srdef1(nf, nf, L)
   H = ( Omega/2 )*H_Rabi + Delta*H_Detuning + H_int + C_def*H_def

   bath = (spa.expm( - beta * H_th))/( spa.expm(-beta * H_th ).diagonal().sum().real )
   rho_in = sps.kron(red, bath)
   rho = rho_in.toarray()

   e, U  = LA.eigh(H.toarray())

   rho_tilde = np.conj(U.T) @ rho @ U
   pa_ti = np.array([pa.toarray() for pa in paulis2])
   pauli2_tilde = np.conj(U.T) @ pa_ti @ U
   model.load_state_dict(torch.load(  './trained_models/PBC/Thermal/eff__Cdef_%s_beta_%s_L_%s_two_bodies_LL' %(C_def, beta,L)))
   logfile = open('./results/test2/evolution_of_two_bodies_Cdef_%s_beta_%s_two_bodies' %(C_def, beta), "w")
   observables_dict_test={}




   for i in range(len(names_pauli2)):
        observables_dict_test[names_pauli2[i]]=((paulis2[i].dot(rho_in).diagonal().sum())/ 
                                           rho_in.diagonal().sum()).real


   v_in = torch.tensor([ observables_dict_test[key] for key in observables_dict_test.keys()  ][1:]).float()
   temp1 = [v_in.numpy()]
   time = [ [0] ]
   temp = [ [ observables_dict_test[key] for key in observables_dict_test.keys()  ][1:] ]
   tr_inter = 0
   eu_inter = 0 
   tr_extra = 0  
   eu_extra = 0  
   for i in range( 1, nstep ):
     time +=  [ [ i * dt ] ]
     ee = np.exp(-1j*e*dt*i)
     rho_in = ee.reshape(ee.shape[0],1)*rho_tilde*np.conj(ee)
     rho_in = rho_in
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
       v_exact = np.array(obs) 
       rho_temp =  np.append(1,v_in.numpy()).reshape(4,4)
       for m in range(4):
        for n in range(4):
         r_recon += 0.25*rho_temp[m][n]*np.kron(pauli_dict[names[m]],pauli_dict[names[n]])
         r_exact += 0.25*v_exact.reshape(4,4)[m][n]*np.kron(pauli_dict[names[m]],pauli_dict[names[n]])
       rho_s_net = 0.5*(Id + v_in.numpy()[:3][0] * sigma_x + v_in.numpy()[:3][1]*sigma_y+v_in.numpy()[:3][2]*sigma_z)
       rho_s_ext = 0.5*(Id + v_in.numpy()[:3][0] * sigma_x + v_in.numpy()[:3][1]*sigma_y+v_in.numpy()[:3][2]*sigma_z)
       r_recon /= r_recon.trace()
       r_exact /= r_exact.trace()
       Lambda,_ = LA.eigh(r_recon-r_exact)
       Lambda_s,_ = LA.eigh(rho_s_net-rho_s_ext)
       if i <= int(nstep/2):
        tr_inter  += np.sum(np.abs(Lambda))
        eu_inter  += distance.euclidean( v_in.numpy()-v_exact[1:]) 
        eu_inter_red    += distance.euclidean( v_in.numpy()[:3]-v_exact[1:][:3])
        tr_inter_red    += np.sum(np.abs(Lambda_s))
       if i > int(nstep/2):
        tr_extra  += np.sum(np.abs(Lambda))
        eu_extra  += distance.euclidean( v_in.numpy()-v_exact[1:])  
        eu_inter_red    += distance.euclidean( v_in.numpy()[:3]-v_exact[1:][:3])
        tr_inter_red    += np.sum(np.abs(Lambda_s)) 
   diff1 =2* tr_inter/nstep
   diff2 =2* eu_inter/nstep
   diff1_s =2* eu_inter_red/nstep
   diff2_s =2* tr_inter_red/nstep
   with open("diff_traj_{sub_n}_PBC_inter_1.txt", "a") as file_object:
     file_object.write("pot %f beta %f tr_d/nstep %f eu_d/nstep %f\n"%( C_def,  beta, diff1,diff2,diff1_s,diff2_s ))
   diff1 =2* tr_extra/nstep
   diff2 =2* eu_extra/nstep
   diff1_s =2* eu_inter_red/nstep
   diff2_s =2* tr_inter_red/nstep
   with open("diff_traj_{sub_n}_PBC_extra_1.txt", "a") as file_object:
     file_object.write("pot %f beta %f tr_d/nstep %f eu_d/nstep %f\n"%( C_def,  beta, diff1,diff2,diff1_s,diff2_s)) 





