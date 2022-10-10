from ml.utils import update_params_from_cmdline, save_metrics_params, ensure_dir, ensure_empty_dir, infinite_dataset
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

## Parameters
L = sub_n

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
                           'beta': [ sub_b],#0., 0.1, 0.3, 0.8, 1., 1.5, 2., 5., 8., 10.],
                           'potential': [sub_c],#0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0],
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
#model_trace = MLP(**conf_deep['MLP_par']).to(conf['device'])
#we load the saved parameters for 1 site impurity
#model.load_state_dict(torch.load('./trained_models/PBC/Thermal/eff__Cdef_%s_beta_%s' %(C_def, beta))) 

#Load the model for extended (2 sites) impurity
#model_trace.load_state_dict(torch.load('./trained_models/PBC/Thermal/eff__Cdef_%s_beta_%s_trace_two_bodies_trace_loss_deep' %(C_def, beta))) #we load the saved parameters


###### DYNAMICS ######


#der = 0
Matr = np.zeros(( len(conf['potential'])  , len(conf['beta'])  ) )
 
for pot in range(len(conf['potential'])):
 for inv_T in range(len(conf['beta'])):
   paulis2, names_pauli2 = pauli_op(sites,L)
   tr_d=0
   C_def = conf['potential'][pot]
   beta = conf['beta'][inv_T] 

   H_Rabi = H1f(sigma_x, L)
   H_Detuning = H1f(nf, L)
   H_int = H3srdef(nf, nf, L)
   H_def = H3srdef1(nf, nf, L)
   H = ( Omega/2 )*H_Rabi + Delta*H_Detuning + H_int + C_def*H_def

#   H_Rabi = H1f(sigma_x, L)
#   H_Detuning = H1f(nf, L)
#   H_sr = H2srdef(nf, nf, L)#   H_sr1 = H1srdef(nf, nf, L)
#   H = ( Omega/2 )*H_Rabi + Delta*H_Detuning + C*H_sr + C_def*H_sr1 
   U = spa.expm( -1j * H * dt)
   U_dag = spa.expm( 1j * H * dt)
   model.load_state_dict(torch.load(  './trained_models/PBC/Thermal/eff__Cdef_%s_beta_%s_L_%s_two_bodies_LL' %(C_def, beta,L)))
   #model.load_state_dict(torch.load('./trained_models/PBC/Thermal/eff__Cdef_%s_beta_%s_two_bodies' %(C_def, beta))) #we load the saved parameters
   logfile = open('./results/test2/evolution_of_two_bodies_Cdef_%s_beta_%s_two_bodies' %(C_def, beta), "w")
   bath = (spa.expm( - beta * H_th))/( spa.expm(-beta * H_th ).diagonal().sum().real )
   rho_in = sps.kron(red, bath)
   #names_pauli2=[]
   #paulis2=[]
   #for i in names:
   #     for j in names:
   #         #names_pauli2.append(i+str(sites[0])+'I'+str(sites[1]))
   #         #paulis2.append(one_F(pauli_dict[i],sites[0],L))
   #         names_pauli2.append(i+str(sites[0])+j+str(sites[1]))
   #         paulis2.append(two_FF(pauli_dict[i],pauli_dict[j],sites[0],sites[1],L))
   #print(names_pauli2   )         
   observables_dict_test={}
   for i in range(len(names_pauli2)):
        observables_dict_test[names_pauli2[i]]=((paulis2[i].dot(rho_in).diagonal().sum())/ 
                                           rho_in.diagonal().sum()).real
   v_in = torch.tensor([ observables_dict_test[key] for key in observables_dict_test.keys()  ][1:]).float()
   temp1 = [v_in.numpy()]
   time = [ [0] ]
   temp = [ [ observables_dict_test[key] for key in observables_dict_test.keys()  ][1:] ]
   for i in range( 1, nstep ):
      time +=  [ [ i * dt ] ]
      rho_in = ((U.dot(rho_in)).dot(U_dag) ) / (rho_in.diagonal().sum())
      #Evolution with Exact diagonalization
      for j in range(len(names_pauli2)):
          observables_dict_test[names_pauli2[j]]=((paulis2[j].dot(rho_in).diagonal().sum())/ 
                                                 rho_in.diagonal().sum()).real
      rho_exact = np.array([observables_dict_test[key] for key in observables_dict_test.keys()  ])
      temp.append( [ observables_dict_test[key] for key in observables_dict_test.keys()  ][1:] )
      #Evolution of the model
      with torch.no_grad():
        data_in = torch.tensor(time[i])
        v_out = model(v_in.float())
        v_in = v_out
        #Check if the matrix obtained is actually positive
        r = np.zeros(16,dtype=complex).reshape(4,4)
        rho_exact = rho_exact.reshape(4,4)
        for m in range(4): 
         for n in range(4): 
          rho_temp =  np.append(1,v_in.numpy()).reshape(4,4)
          r += 0.25*rho_temp[m][n]*np.kron(pauli_dict[names[m]],pauli_dict[names[n]])
        r /= r.trace()
        print(r.trace())
        #projection over the density matrix set
        sigma,Q = LA.eigh(r) 
        for s in sigma:
              if s<0: print("reconstructed rho is non positive")
        sigma   = [ max(s,0) for s in sigma]
        sigma  /= np.sum(sigma)
        rt = np.einsum('ik,k,kl->il',Q,sigma,np.conj(Q.T))
        landa,_ = LA.eigh(r-rho_exact.reshape(4,4))
        eu_d += np.sum(np.abs(r.reshape(16)-rho_exact.reshape(16) ) )  
        tr_d  += np.sum(np.abs(landa))
        print('Time %f'%(i*dt ) )
        for namez in names_pauli2:
                        observables_dict_test[namez]=((np.kron(pauli_dict[namez[0]],pauli_dict[namez[2]])\
                        .dot(rt).diagonal().sum())/ rt.diagonal().sum()).real
        temp1.append( [ observables_dict_test[key] for key in observables_dict_test.keys()  ][1:] )
        #temp1.append(v_in.numpy())
   diff1 = tr_d/nstep
   diff2 = eu_d/nstep
   #Matr[pot, inv_T] = diff 
   with open("diff_sub_n_spin_chain_final.txt", "a") as file_object:
     file_object.write(" %d %d pot %f beta %f tr_d/nstep %f eu_d/nstep %f\n"%(sub_it_c, sub_it_b, C_def,  beta, diff1,diff2)) #Matr[pot,inv_T] = tr_d/nstep
"""
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
ax.set_ylabel(names_pauli2[i+1])
plt.show()
plt.matshow(Matr)
plt.show()
plt.savefig('diff.png')
"""

#np.savetxt('dff.txt', Matr)
