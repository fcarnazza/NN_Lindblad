import scipy.sparse.linalg as spa
import scipy.sparse as sps
from  ml.utils import update_params_from_cmdline, save_metrics_params, ensure_dir
import os
from physics.FSS_dyn.ED_b import *
import deepdish as dd
from scipy import linalg as LA

default_parameters = {'model_dir': './results/data_th_periodic/T_10',
                      'system': {
                          'n_spins': 1,  # system size
                           'potential': [0.4],
                           'Rabi': 1,
                           'Delta': 0,
                           'gamma': [0.06],

                      },
                      'dt': 0.01,  # time step
                      'time': 10,  # total time of the evolution
                      'num_trajectories': 30,  # number
                      'seed': None
                      }

#	**********  BASIC MATRIX DEFINITIONS  **********
#
sigma_x = np.array( [ [ 0. , 1. ] , [ 1. , 0. ] ] )
N = np.array( [ [ 0. , 0. ] , [ 0. , 1. ] ] )
sigma_z = np.array( [ [ 1. , 0. ] , [ 0. , -1. ] ] )
Id = np.array( [ [ 1. , 0. ] , [ 0. , 1. ] ] )
sigma_y = np.array([[0, -1j], [1j, 0]])
Pj = np.array( [ [ 1. , 0. ] , [ 0. , 0. ] ] )
nf = np.array([[1, 0],[0, 0]]) ###  local fermions number nf = 1-sz ###
sigma_m = np.array([[0,0],[1,0]] )
pauli_dict = {
    'X': sigma_x,
    'Y': sigma_y,
    'Z': sigma_z,
    'I': Id
}
names=['I','X','Y','Z']


def compute_traj2( sites, dt, T):
    #Let us explicitly create in Schroedinger picture the Lindblad operator 
    #   .                           1            +      1   +
    # \rho = -i [H,\rho] + \gamma \sum  L  \rho L     - - {L L ,  \rh0 }
    #                             i = 1  k        k     2   k k
    names_pauli2=[]
    paulis2=[]
    observables_dict={}
    nstep = int(T / dt)
    H_Rabi = H1f(sigma_x,  1)
    rho_loc=np.array([np.random.normal() for i in range(4)]).reshape(2,2)+1j*np.array([np.random.normal() for i in range(4)]).reshape(2,2)
    rho_loc=rho_loc.dot(rho_loc.conjugate().transpose())
    rho_loc=rho_loc/rho_loc.trace()
    rho_in  = rho_loc
    Liouville = Liouvillian(H , [J],1,gamma)
    for key in names:
        observables_dict[key]=((pauli_dict[key].dot(rho_in).diagonal().sum())/ 
                                           rho_in.diagonal().sum()).real
    arr = [[ observables_dict[key] for key in observables_dict.keys()  ][1:]]
    U = spa.expm(  Liouville * dt)
    temp = [ 0.0 ]
    for i in range( 1, nstep ):
        rho_in = rho_in.reshape(4,1)
        rho_in = U @ rho_in
        vals_L, _ = LA.eig(Liouville.toarray())
        rho_in = rho_in.reshape(2,2)
        rho_in = rho_in/rho_in.diagonal().sum()
        for key in names:
         observables_dict[key]=((pauli_dict[key].dot(rho_in).diagonal().sum())/ 
                                           rho_in.diagonal().sum()).real
        o = [ observables_dict[key] for key in observables_dict.keys()  ]
        arr.append( o[1: ] )
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
            C_def = cfg["system"]["potential"][k]
            Delta = cfg["system"]["Delta"]
            gamma = cfg["system"]["gamma"][k]

            print("----Generating trajectories for impurity = %s" % C_def)
            sites = [0]
            H_Rabi = H1f(sigma_x, L)
            T = cfg["time"]
            dt = cfg["dt"]
            nstep = int(T/dt)
            time = np.linspace(0 , T-2*dt , nstep - 1 , endpoint = True)
            #Jump operators 
            J = sigma_m
            #Effective Hamiltonian                 
            H = Omega*H_Rabi 


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

            path = os.path.join(cfg["model_dir"], f"traj_fss_dyn_gamma_{gamma}_Lindblad_diag_1.h5")
            path_eval_fin = os.path.join(cfg["model_dir"], f"traj_fss_final_gamma_{gamma}_Lindblad_diag_1.h5")

            print(f"saving to {path}")
            dd.io.save(path, trajectories)
            dd.io.save(path_eval_fin, trajectories_eval_fin)
            print("done")
            metrics = {'dummy': 0}
