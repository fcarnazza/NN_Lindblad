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
import matplotlib
from get_Lindblad import get_L_from_model, get_omega_from_model

# Parameters of the header
conf = {'device': 'cpu',
        'beta':[0.0, 0.1, 0.3, 0.8, 1.0, 1.5, 2.0, 5.0, 8.0, 10.0],
        'potential':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0],
        'Rabi':  1,
        'Delta': 0,
        'MLP_par': {
                'mlp_params': {
                        'data_dim': 15,
                        'layers':  [ ],
                        'nonlin': 'id',
                        'output_nonlin': 'id'
                        },
                }
        }



sigma_x = np.array( [ [ 0. , 1. ] , [ 1. , 0. ] ] )
nf = np.array([[1, 0],[0, 0]]) ###  local fermions number nf = 1-sz ###
 
l_shift = np.zeros(len(conf['potential'])*len(conf['beta'])).reshape( len(conf['potential']), len(conf['beta'])  )
sites = [1,2]
for L_tot in np.arange(7,13,2):
  for C_def_idx,C_def in enumerate(conf['potential']):
   for beta_idx, beta in enumerate(conf['beta']):
     Omega = conf["Rabi"]
     Delta = conf["Delta"]
     L = 2 
     paulis2, names_pauli2 = pauli_op(sites,L)
       
     H_Rabi = H1f(sigma_x, L)
     H_Detuning = H1f(nf, L)
     H_int = H3srdef(nf, nf, L)
     H_def = H3srdef1(nf, nf, L)
     H = ( Omega/2 )*H_Rabi + H_int + C_def*H_def
     H = H.todense()
     omega = get_omega_from_model( './trained_models/PBC/Thermal/eff__Cdef_%s_beta_%s_L_%s_two_bodies_LL' %(C_def, beta,L_tot)  )
     recon_H = np.zeros(16).reshape(4,4)
     for m in range(len(omega)):
             recon_H = recon_H + paulis2[1:][m]*omega[m]
     H_basis = np.zeros(16).reshape(4,4)
     for m in range(len(omega)):
             H_basis = H_basis + paulis2[1:][m]*np.trace(paulis2[1:][m]@H)*0.25
     # This is because the Hilbert space dimension is 4
     H_basis = 0.25*H_basis
     shift = H_basis - recon_H
     l_shift[C_def_idx][beta_idx] = np.sqrt(np.trace(shift.T@shift))
     if(C_def==0.1  ):
          print(H)
          print(recon_H)
          print(H_basis)
          H_basis = np.real(H_basis)
          fig, ax = plt.subplots()
          cax = ax.matshow( H_basis)#,interpolation='nearest')
          norm = matplotlib.colors.Normalize(vmin=min(np.array(H_basis).flatten()), vmax=max(np.array(H_basis).flatten()))
          ax.imshow(H_basis, cmap='Greys',norm=norm)#,  interpolation='nearest')
          cmap = matplotlib.cm.binary
          fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
          plt.savefig('./images/H_PBC_pot_0.1_beta_%s.pdf'%beta)

          fig, ax = plt.subplots()
          cax = ax.matshow( np.real(recon_H))#,interpolation='nearest')
          norm = matplotlib.colors.Normalize(vmin=min(np.array(H_basis).flatten()), vmax=max(np.array(H_basis).flatten()))
          ax.imshow(np.real(recon_H), cmap='Greys',norm=norm)#,  interpolation='nearest')
          cmap = matplotlib.cm.binary
          fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
          plt.savefig('./images/H_recon_PBC_pot_0.1_beta_%s.pdf'%beta)

          fig, ax = plt.subplots()
          cax = ax.matshow(np.real(shift) )#,interpolation='nearest')
          norm = matplotlib.colors.Normalize(vmin=min(np.array(H_basis).flatten()), vmax=max(np.array(H_basis).flatten()))
          ax.imshow(np.real(shift), cmap='Greys',norm=norm)#,  interpolation='nearest')
          cmap = matplotlib.cm.binary
          fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
          plt.savefig('./images/diff_pot_0.1_beta_%s.pdf'%beta)

  ind = np.arange(10)
  fig, ax = plt.subplots()
  ax.set_xticks(ind)
  ax.set_yticks(ind)
  cax = ax.matshow(l_shift)#,interpolation='nearest')
  ax.set_xticklabels( conf['potential']  )
  ax.set_yticklabels( conf['beta']  )
  ax.set_ylabel('beta')
  ax.set_xlabel('pot')
  norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
  ax.imshow(l_shift, cmap='Greys',norm=norm)#,  interpolation='nearest')
  cmap = matplotlib.cm.binary

  fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
  plt.savefig('./images/L_shift_PBC_%d.pdf'%(L_tot))
