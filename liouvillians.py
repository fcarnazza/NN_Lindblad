from ml.utils import update_params_from_cmdline, save_metrics_params, ensure_dir, ensure_empty_dir, infinite_dataset
from torch.utils.data import Dataset, DataLoader
from ml.pytorch_modules.vae import BVAE
from ml.pytorch_modules.dynamics_mlp import MLP
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
import sys
## Parameters
L = 7

dt = 0.01
T = 20 
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
                           'beta': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                           'potential': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
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
sigma=np.array([Id,sigma_x,sigma_y,sigma_z])

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

conf = default_parameters_test
model = MLP(**conf['MLP_par']).to(conf['device'])#we load the blueprin#t of the model



###### DYNAMICS ######


Matr = np.zeros(( len(conf['potential'])  , len(conf['beta'])  ) )
 
tr_d=0
C_def = sys.argv[1]
beta =  sys.argv[2] 
C_def = float(C_def)/10
beta = float(beta)/10
print(beta)
H_Rabi = H1f(sigma_x, L)
H_Detuning = H1f(nf, L)
H_int = H3srdef(nf, nf, L)
H_def = H3srdef1(nf, nf, L)
H = ( Omega/2 )*H_Rabi + Delta*H_Detuning + H_int + C_def*H_def
U = spa.expm( -1j * H * dt)
U_dag = spa.expm( 1j * H * dt)
model.load_state_dict(torch.load(  './trained_models/PBC/Thermal/eff__Cdef_%s_beta_%s_L_%s_two_bodies' %(C_def, beta,L)))
#logfile = open('./results/test2/evolution_of_two_bodies_Cdef_%s_beta_%s_two_bodies' %(C_def, beta), "w")
bath = (spa.expm( - beta * H_th))/( spa.expm(-beta * H_th ).diagonal().sum().real )
rho_in = sps.kron(red, bath)
names_pauli2=[]
paulis2=[]
for i in names:
     for j in names:
         names_pauli2.append(i+str(sites[0])+j+str(sites[1]))
         paulis2.append(two_FF(pauli_dict[i],pauli_dict[j],sites[0],sites[1],L))
#print(names_pauli2   )         
observables_dict_test={}
for i in range(len(names_pauli2)):
     observables_dict_test[names_pauli2[i]]=((paulis2[i].dot(rho_in).diagonal().sum())/ 
                                        rho_in.diagonal().sum()).real




matr = np.eye(16)
for name, param in model.named_parameters():
                if 'weight' in name:
                         matr[1:,1:] = param.data.numpy() @ matr[1:,1:]
                         plt.matshow(param.data.numpy() )
                         plt.show( )
                         matr[1:,:1] = param.data.numpy() @ matr[1:,:1]
                if 'bias'  in name:
                         matr[1:,:1]  = matr[1:,:1] + param.data.numpy().reshape(15,1)
matr = (matr - np.eye(16))/0.01
gap = np.sort(np.abs(LA.eigvals(matr).real))[1]

#with open("gaps.txt", "a") as f:
#     f.write("beta %f, C_def %f, gap %f \n"%(beta, C_def, gap))

eig_diff = np.abs(LA.eigvals((matr + matr.transpose())/2.)).sum()

with open("Lio.txt", "a") as f:
     f.write("beta %f, C_def %f, sym part%f \n"%(beta, C_def,eig_diff ))






