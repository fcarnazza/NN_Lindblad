import scipy.sparse.linalg as spa
import scipy.sparse as sps
from ml.utils import update_params_from_cmdline, save_metrics_params, ensure_dir, xmap
import os
from physics.FSS_dyn.ED_b import *
import deepdish as dd
#import numba
from s_const import *
from multiprocessing import Pool, cpu_count
import opt_einsum as oe
import scipy.linalg as LA
default_parameters = {'model_dir': './results/data_th_open/T_10',
                      'system': {
                          'n_spins': 6,  # system size (choose even)
                           'alpha' : [0.5,3.0],
                           'potential': [0.1,0.5],
                           'Rabi': 1,
                           'Delta': 0,

                      },
                      'dt': 0.005,  # time step
                      'time': 10,  # total time of the evolution
                      'num_trajectories': 20,  # number
                      'seed': None
                      }

#   **********  BASIC MATRIX DEFINITIONS  **********
#
sigma_x = np.array( [ [ 0. , 1. ] , [ 1. , 0. ] ] )
N = np.array( [ [ 0. , 0. ] , [ 0. , 1. ] ] )
sigma_z = np.array( [ [ 1. , 0. ] , [ 0. , -1. ] ] )
sigma_y = np.array([[0, -1j], [1j, 0]])
Pj = np.array( [ [ 1. , 0. ] , [ 0. , 0. ] ] )
nf = np.array([[1, 0],[0, 0]]) ###  local fermions number nf = 1-sz ###
Id = np.eye(2).astype(np.float64)
observables_dict={}

def compute_traj2(sites,  dt, T):
    paulis2, names_pauli2 = pauli_op(sites,l)
    nstep = int(T / dt)
    bath = sps.eye(2**((L-2)/2))
    rho_loc=np.array([np.random.normal() for i in range(16)]).reshape(4,4)+1j*np.array([np.random.normal() for i in range(16)]).reshape(4,4)
    rho_loc=rho_loc.dot(rho_loc.conjugate().transpose())
    rho_loc=rho_loc/rho_loc.trace()
    
    rho = np.kron(bath.toarray(),np.kron(rho_loc, bath.toarray()))
    for i in range(len(names_pauli2)):
        observables_dict[names_pauli2[i]]=((paulis2[i].dot(rho).diagonal().sum())/ 
                                           rho.diagonal().sum()).real

    arr = [[ observables_dict[key] for key in observables_dict.keys()  ][1:]]
    print(arr)
    temp = [ 0.0 ]
    rho_tilde = np.conj(U.T) @ rho @ U
    pa_ti = np.array([pa.toarray() for pa in paulis2])
    pauli2_tilde = np.conj(U.T) @ pa_ti @ U
    for i in range( 1, nstep ):
        ee = np.exp(-1j*e*dt*i)
        rho_in = ee.reshape(ee.shape[0],1)*rho_tilde*np.conj(ee)
        rho_in = rho_in

        trace = lambda k: ((oe.contract( 'ij,ji->', pauli2_tilde[k],rho_in ))/ rho_in.diagonal().sum()).real

        obs = xmap(trace,range(len( names_pauli2 ) ))
        time = i * dt
        arr.append( obs[1:] ) #observables_dict[key] for key in observables_dict.keys()  ][1:] )
        temp.append( i * dt )
    print("finished Time = %s" % T)
    return np.asarray(arr[:-1]).astype(np.float32), \
           np.asarray(arr[1:]).astype(np.float32), \
           np.asarray(temp[:-1]).astype(np.float32), \
           np.expand_dims(np.asarray(arr[0]).astype(np.float32), 0), \
           np.expand_dims(np.asarray(arr[-1]).astype(np.float32), 0)


def compute_traj(site, dt, T):
    nstep = int(T / dt)
    a = np.random.uniform(0,1)
    x = np.sqrt(np.random.uniform(0, a))
    y = np.sqrt(a - x**2) 
    rho_loc = 0.5*np.array( [ [ 1 , x + 1j*y ],  [   x - 1j*y, 1  ] ] )
    red = sps.csc_matrix(rho_loc)
    bath = sps.eye(2**((L-1)/2))
    rho_in = sps.kron(bath, sps.kron(red, bath))
    O0 = one_F(Id, site, L)
    O1 = one_F(sigma_x, site, L)
    O2 = one_F(sigma_y, site, L)
    O3 = one_F(sigma_z, site, L)
    OId = ((O0.dot(rho_in).diagonal().sum())/ rho_in.diagonal().sum()).real
    Ox =  ((O1.dot(rho_in).diagonal().sum())/ rho_in.diagonal().sum()).real 
    Oy =  ((O2.dot(rho_in).diagonal().sum())/ rho_in.diagonal().sum()).real 
    Oz =  ((O3.dot(rho_in).diagonal().sum())/ rho_in.diagonal().sum()).real 
    arr = [ [OId, Ox, Oy, Oz ]  ]
    print(arr)
    U = spa.expm( -1j * H * dt)
    U_dag = spa.expm( 1j * H * dt)
    temp = [ 0.0 ]
    for i in range( 1, nstep ):
        rho_in = ((U.dot(rho_in)).dot(U_dag) ) / (rho_in.diagonal().sum())
        Ox =  ((O1.dot(rho_in).diagonal().sum())/ rho_in.diagonal().sum()).real 
        Oy =  ((O2.dot(rho_in).diagonal().sum())/ rho_in.diagonal().sum()).real 
        Oz =  ((O3.dot(rho_in).diagonal().sum())/ rho_in.diagonal().sum()).real
        time = i * dt  
        arr.append( [OId, Ox, Oy, Oz ] )
        temp.append( i * dt )
    print("finished Time = %s" % T)
    return np.asarray(arr[:-1]).astype(np.float32), \
           np.asarray(arr[1:]).astype(np.float32), \
           np.asarray(temp[:-1]).astype(np.float32), \
           np.expand_dims(np.asarray(arr[0]).astype(np.float32), 0), \
           np.expand_dims(np.asarray(arr[-1]).astype(np.float32), 0)

if __name__ == '__main__':
    
    cfg = default_parameters
    L = cfg["system"]["n_spins"]
    Omega = cfg["system"]["Rabi"]
    for k in range(len(cfg["system"]["potential"])):
        for j in range(len(cfg["system"]["alpha"])):
            C = cfg["system"]["potential"][k]
            Delta = cfg["system"]["Delta"]
            alpha = cfg["system"]["alpha"][j]


            print("----Generating trajectories for interaction strength = %s and for interaction range = %s" % (C, alpha))


            H_Rabi = H1f(sigma_x, L)
            H_Detuning = H1f(nf, L)
            H_lr_ferm = H2fflr(nf, nf, L, alpha )
            T = cfg["time"]
            dt = cfg["dt"]
            nstep = int(T/dt)
    

            sites = [int(L/2),int(L/2)+1]

    
            H = ( Omega/2 )*H_Rabi + Delta*H_Detuning + C*H_lr_ferm
    
            e, U = LA.eigh(H.toarray())
            print("diagonalized")


            ensure_dir(cfg["model_dir"])

            if cfg["seed"] is not None:
                np.random.seed(cfg["seed"])

            trajectories = {}
            trajectories['p_t1'] = None
            trajectories['p_t2'] = None
            trajectories['t'] = None

            trajectories_eval_fin = {}
            trajectories_eval_fin['p_0'] = None
            trajectories_eval_fin['p_T'] = None

            for i in range(cfg["num_trajectories"]):
                print(f"trajectory: {i}")
                random_state= np.random.randint(0, 2**L)
                p_t1, p_t2, t, p_0, p_T = compute_traj2(sites, cfg["dt"], cfg["time"])

                if trajectories['p_t1'] is None:
                    trajectories['p_t1'] = p_t1
                    trajectories['p_t2'] = p_t2
                    trajectories['t'] = t
                    trajectories_eval_fin['p_0'] = p_0
                    trajectories_eval_fin['p_T'] = p_T
                else:
                    trajectories['p_t1'] = np.concatenate((trajectories['p_t1'], p_t1), 0)
                    trajectories['p_t2'] = np.concatenate((trajectories['p_t2'], p_t2), 0)
                    trajectories['t'] = np.concatenate((trajectories['t'], t), 0)
                    trajectories_eval_fin['p_0'] = np.concatenate((trajectories_eval_fin['p_0'], p_0), 0)
                    trajectories_eval_fin['p_T'] = np.concatenate((trajectories_eval_fin['p_T'], p_T), 0)

            path = os.path.join(cfg["model_dir"], f"traj_alpha_{alpha}_C_{C}_n_spin_{L}_dt_{dt}.h5")
            path_eval_fin = os.path.join(cfg["model_dir"], f"traj_final_alpha_{alpha}_C_{C}_n_spins_{L}_dt_{dt}.h5")

            print([v.shape for k, v in trajectories.items()])
            print(f"saving to {path}")
            dd.io.save(path, trajectories)
            dd.io.save(path_eval_fin, trajectories_eval_fin)
            print("done")
            metrics = {'dummy': 0}
