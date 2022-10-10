from ml.utils import update_params_from_cmdline, save_metrics_params, ensure_dir, ensure_empty_dir, infinite_dataset
from torch.utils.data import Dataset, DataLoader
from ml.pytorch_modules.vae import BVAE
from ml.pytorch_modules.dynamics_mlp import MLP
from torch import optim
from train_data import *
from utils.settings import cfg
import os
from tensorboardX import SummaryWriter #This module enable to monitor, live, the training later we will show
import numpy as np
import matplotlib.pyplot as plt
from physics.FSS_dyn.ED_b import *
import scipy.sparse.linalg as spa
import scipy.sparse as sps
from scipy import linalg as LA

## Parameters
dt = 0.01
T = 20
nstep = int(T/dt)
Omega = 1.0
Delta = 0
C_def = 0.5
sites = [1,2]
gamma = 0.05
L = 2
sigma_x = np.array( [ [ 0. , 1. ] , [ 1. , 0. ] ] )
N = np.array( [ [ 0. , 0. ] , [ 0. , 1. ] ] )
sigma_z = np.array( [ [ 1. , 0. ] , [ 0. , -1. ] ] )
sigma_y = np.array([[0, -1j], [1j, 0]])
Pj = np.array( [ [ 1. , 0. ] , [ 0. , 0. ] ] )
nf = np.array([[1, 0],[0, 0]]) ###  local fermions number nf = 1-sz ###
Id = np.eye(2).astype(np.float64)

sigma=np.array([Id,sigma_x,sigma_y,sigma_z])
sigma_m = np.array([[0,0],[1,0]] )
pauli_dict = {
'X': sigma_x,
'Y': sigma_y,
'Z': sigma_z,
'I': Id
}
names=['I','X','Y','Z']

# Parameters of the header
default_parameters_test = {'device': 'cpu',
              'MLP_par': {
                        'mlp_params': {
                                'data_dim': 15,
                                'layers':  [ 15,15],
                      'nonlin': 'id',
                      'output_nonlin': 'id'
                                                        },
                  },
              }


#### Initial state ###
#rho1 = np.array( [ [ 0.8 , 0 ],  [  0 , 0.2  ] ] ) #State of the sub-system
#rho2 = np.array( [ [ 0.4 , 0 ],  [  0 , 0.6  ] ] )
rho_loc=np.array([np.random.normal() for i in range(16)]).reshape(4,4)+1j*np.array([np.random.normal() for i in range(16)]).reshape(4,4)
rho_loc=rho_loc.dot(rho_loc.conjugate().transpose())
rho_loc=rho_loc/rho_loc.trace()

rho_in  = sps.csc_matrix(rho_loc)
#H_def = C_def*two_FF(nf, nf, 1, 2, L)
H_def = C_def*sps.kron(nf,nf)
two_FF(nf, nf, 1, 2, L)
H_Rabi = H1f(sigma_x,  2)

sites = [0,1]
H = Omega*H_Rabi + H_def 
J = np.array([np.kron(np.kron(np.eye(j),sigma_m),np.eye(L-j+1)) for j in range(1,L+1)]) 


Liouville = -1j * np.kron(H ,np.eye(L) ) + 1j * np.kron(np.eye(L),H) 
Liouville = Liouvillian(H, J, L, gamma)
print("HAMILTONIAN BUILT")

#OBSERVABLES 
names_pauli2=[]
paulis2=[]
observables_dict={}
for i in names: 
        for j in names:
                sigma_i = sps.csc_matrix(pauli_dict[i])
                sigma_j = sps.csc_matrix(pauli_dict[j])
                names_pauli2.append(i+str(sites[0])+j+str(sites[1]))
                paulis2.append( sps.kron(sigma_i,sigma_j))
for i in range(len(names_pauli2)):
        observables_dict[names_pauli2[i]]=((paulis2[i].dot(rho_in).diagonal().sum())/ 
                               rho_in.diagonal().sum()).real
arr = [[ observables_dict[key] for key in observables_dict.keys()  ]]
print(arr)
#Lindblad superop
U = spa.expm(  Liouville * dt)

#if default_parameter_test['MLP_par']['mlp_params']['data_dim']%2 == 0: 
#v_in = torch.tensor([ observables_dict[key] for key in observables_dict.keys()  ]).float()#[1:]).float()
v_in = torch.tensor([ observables_dict[key] for key in observables_dict.keys()  ][1:]).float()
print(v_in)
temp1 = [v_in.numpy()]

conf = default_parameters_test
model = MLP(**conf['MLP_par']).to(conf['device'])#we load the blueprint of the model

#Load the model for extended (2 sites) impurity
model.load_state_dict(torch.load('./trained_models/PBC/Thermal/eff__Cdef_%s_two_bodies_Lindblad_deep'%(C_def))) #we load the saved parameters




###### DYNAMICS ######
matr = np.eye(16)
for name, param in model.named_parameters():
                if 'weight' in name:
                         matr[1:,1:] = param.data.numpy() @ matr[1:,1:] 
                         matr[1:,:1] = param.data.numpy() @ matr[1:,:1]
                if 'bias'  in name:
                         matr[1:,:1]  = matr[1:,:1] + param.data.numpy().reshape(15,1)
#matr[:,0] = matr[:,0] + o

#plt.matshow((matr-np.eye(16))/0.01)
#plt.show()

Li=Liouville.toarray().reshape(2,2,2,2,2,2,2,2)
Li=0.25*np.einsum('aki,blj,ijklpqrs,mpr,nqs',sigma,sigma,Li,sigma,sigma)
Li = Li.reshape(16,16)
Li = Li.real
#plt.matshow(Li)
#plt.show( )

Matr = (matr - np.eye(16))/0.01
print("diff = %f"%np.absolute(LA.eigvals(Matr-Li)).sum())
time = [ [0] ]
Li = Li.reshape(16,16).real
Li = spa.expm(sps.csc_matrix(Li) *dt )
#Li = Li * 0.01 + sps.eye(16)
rho_in =  np.array( [ observables_dict[key] for key in observables_dict.keys()  ] )


temp = [ [ observables_dict[key] for key in observables_dict.keys()  ][1:] ]
for i in range( 1, nstep ):

 #print( 'Time = %.2f ' % (i * dt) )
 time +=  [ [ i * dt ] ]
 rho_in = rho_in.reshape(16,1)
 rho_in = Li @ rho_in
 #rho_in = matr @ rho_in
 rho_in = rho_in.reshape(4,4)
 #rho_in = rho_in/rho_in.diagonal().sum()
 #for k in range(len(names_pauli2)):
 #   observables_dict[names_pauli2[k]]=((paulis2[k].dot(rho_in).diagonal().sum())/ rho_in.diagonal().sum()).real
 #temp.append( [ observables_dict[key] for key in observables_dict.keys()  ][1:] )

 rho_in = rho_in.reshape(16)
 temp.append( [ exp_val for exp_val in rho_in ][1:] )

    #Evolution of the model
 with torch.no_grad():
      data_in = torch.tensor(time[i])
      v_out = model(v_in.float())
      v_in = v_out
      #Check if the matrix obtained is actually positive
      r = np.zeros(16,dtype=complex).reshape(4,4)
      for m in range(4): 
       for n in range(4): 
        rho_temp = np.append(1,v_in.numpy()).reshape(4,4)
        #rho_temp = v_in.numpy().reshape(4,4)
        r += 0.25*rho_temp[m][n]*np.kron(pauli_dict[names[m]],pauli_dict[names[n]])
      r /= r.trace() 
      print(r.trace())
      #projection over the density matrix set
      sigma,Q = LA.eigh(r) 
      for s in sigma:
            if s<0: print("reconstructed rho is non positive, %f"%s)
      #print(sigma)
#      sigma   = [ max(s,0) for s in sigma]
      sigma  /= np.sum(sigma)
      rt = np.einsum('ik,k,kl->il',Q,sigma,np.conj(Q.T))
      for k in range(len(names_pauli2)):
        observables_dict[names_pauli2[k]]=((paulis2[k].dot(rt).diagonal().sum())/ rt.diagonal().sum()).real
      temp1.append( [ observables_dict[key] for key in observables_dict.keys()  ][1:] )

     
      #temp1.append(v_in.numpy()[1:]/r.trace())


err =[]
for i in range(15):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x3=[x[i] for x in temp]
    x31=[x[i] for x in temp1]
    ax.plot(time, x3, 'r-', linewidth=2)
    ax.plot(time, x31, 'k-.', linewidth=2)
    ax.set_xlabel('t')
    ax.set_ylabel(names_pauli2[i+1])
    plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(time, temp, 'r-', linewidth=2)
ax.plot(time, temp1, 'k-.', linewidth=2)
ax.set_xlabel('t')
ax.set_ylabel(names_pauli2t[i+1])
plt.show()
