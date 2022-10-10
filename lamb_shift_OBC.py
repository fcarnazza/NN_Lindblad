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

conf = {'device': 'cpu',
                           'alpha' :  [0.0, 0.3, 0.5, 1.0, 1.2, 2.0, 3.0, 6.0, 10.0, 20.0], 
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
                                   },
                           }


sigma_x = np.array( [ [ 0. , 1. ] , [ 1. , 0. ] ] )
nf = np.array([[1, 0],[0, 0]]) ###  local fermions number nf = 1-sz ###





l_shift = np.zeros(len(conf['potential'])*len(conf['alpha'])).reshape( len(conf['potential']), len(conf['alpha'])  )
for L_tot in np.arange(6,12,2):
  for C_def_idx,C_def in enumerate(conf['potential']):
   for alpha_idx, alpha in enumerate(conf['alpha']):
     L = L_tot 
     sites = [int(L/2),int(L/2)+1]
     sites = [1,2]
     L = 2 
     Omega = conf["Rabi"]
     Delta = conf["Delta"]
     paulis2, names_pauli2 = pauli_op(sites,L)
     H_Rabi = H1f(sigma_x, L)
     H_Detuning = H1f(nf, L)
     H_lr_ferm = H2fflr(nf, nf, L, alpha )
     print(H_lr_ferm.todense())
     H_tot = ( Omega/2 )*H_Rabi + C_def*H_lr_ferm
     H_basis = np.zeros(16).reshape(4,4)
     #omega_basis = []
     #for k in range(15):
     #        omega_basis = omega_basis + [0.25*np.trace(paulis2[1:][k]@H_tot.todense())/(2**L)]
     #H = H.todense()
     #paulis2, names_pauli2 = pauli_op(sites,l)
     omega = get_omega_from_model( './trained_models/OBC/Thermal/eff__alpha_%s_C_%s_n_spins_%s' %(alpha, C_def,L_tot))  
     recon_H = np.zeros(16).reshape(4,4)
     for m in range(len(omega)):
             recon_H = recon_H + paulis2[1:][m]*omega[m]
          #   H_basis = H_basis + paulis2[1:][m]*omega_basis[m]
     H_basis = np.zeros(16).reshape(4,4)
     for m in range(len(omega)):
             H_basis = H_basis + paulis2[1:][m]*np.trace(paulis2[1:][m]@H_tot.todense())*0.25
     #print(H_basis)
     H_basis = 0.25*H_basis
     shift = H_basis - recon_H
     l_shift[C_def_idx][alpha_idx] = np.sqrt(np.trace(shift.T@shift))
     if(C_def==0.1 ):
          H_basis = np.real(H_basis)
          fig, ax = plt.subplots()
          cax = ax.matshow( H_basis)#,interpolation='nearest')
          norm = matplotlib.colors.Normalize(vmin=min(np.array(recon_H).flatten()), vmax=max(np.array(recon_H).flatten()))
          ax.imshow(H_basis, cmap='Greys',norm=norm)#,  interpolation='nearest')
          cmap = matplotlib.cm.binary
          fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
          plt.savefig('./images/H_OBC_pot_0.1_alpha_%s.pdf'%alpha)

          fig, ax = plt.subplots()
          cax = ax.matshow( np.real(recon_H))#,interpolation='nearest')
          norm = matplotlib.colors.Normalize(vmin=min(np.real(np.array(recon_H)).flatten()), vmax=max(np.real(np.array(recon_H)).flatten()))
          ax.imshow(np.real(recon_H), cmap='Greys',norm=norm)#,  interpolation='nearest')
          cmap = matplotlib.cm.binary
          fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
          plt.savefig('./images/H_recon_OBC_pot_0.1_alpha_%s.pdf'%alpha)

          fig, ax = plt.subplots()
          cax = ax.matshow(np.real(shift) )#,interpolation='nearest')
          norm = matplotlib.colors.Normalize(vmin=min(np.real(np.array(shift)).flatten()), vmax=max(np.real(np.array(shift)).flatten()))
          ax.imshow(np.real(shift), cmap='Greys',norm=norm)#,  interpolation='nearest')
          cmap = matplotlib.cm.binary
          fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
          plt.savefig('./images/diff_pot_0.1_alpha_%s.pdf'%alpha)
          print("Lamb shift for c_def = 0.1 and alpha = %s"%alpha)
          print(np.diag(shift))
          print(H_lr_ferm@shift)
  ind = np.arange(10)
  fig, ax = plt.subplots()
  ax.set_xticks(ind)
  ax.set_yticks(ind)
  cax = ax.matshow(l_shift)#,interpolation='nearest')
  ax.set_xticklabels( conf['potential']  )
  ax.set_yticklabels( conf['alpha']  )
  ax.set_ylabel('alpha')
  ax.set_xlabel('pot')
  norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
  ax.imshow(l_shift, cmap='Greys',norm=norm)#,  interpolation='nearest')
  cmap = matplotlib.cm.binary

  fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
  plt.savefig('./images/L_shift_OBC_%d.pdf'%(L_tot))
