import scipy.sparse.linalg as spa
import scipy.sparse as sps
from ml.utils import update_params_from_cmdline, save_metrics_params, ensure_dir
import os
from physics.FSS_dyn.ED_b import *
import deepdish as dd
import numba
from s_const import *
import opt_einsum as oe 
default_parameters = {'model_dir': './results/data_th_periodic/T_10/two_bodies',
                      'system': {
                          'n_spins': 9,  # system size
                           #'alpha' : [ 6, 2, 0 ],
                           'beta': [0.5],
                           'potential': [0.5],
                           'Rabi': 1,
                           'Delta': 0,

                      },
                      'dt': 0.01,  # time step
                      'time': 10,  # total time of the evolution
                      'num_trajectories': 50,  # number
                      'seed': None
                      }

#       **********  BASIC MATRIX DEFINITIONS  **********
#
sigma_x = np.array( [ [ 0. , 1. ] , [ 1. , 0. ] ] )
N = np.array( [ [ 0. , 0. ] , [ 0. , 1. ] ] )
sigma_z = np.array( [ [ 1. , 0. ] , [ 0. , -1. ] ] )
Id = np.array( [ [ 1. , 0. ] , [ 0. , 1. ] ] )
sigma_y = np.array([[0, -1j], [1j, 0]])
Pj = np.array( [ [ 1. , 0. ] , [ 0. , 0. ] ] )
nf = np.array([[1, 0],[0, 0]]) ###  local fermions number nf = 1-sz ###

pauli_dict = {
    'X': sigma_x,
    'Y': sigma_y,
    'Z': sigma_z,
    'I': Id
}
names=['I','X','Y','Z']

basis = get_basis(4)
#@numba.njit(parallel=True)
def compute_traj2(sites, beta, dt, T):
    names_pauli2=[]
    paulis2=[]
    observables_dict={}
    nstep = int(T / dt)
    H_Rabi_th = H1f(sigma_x, L - 2)
    H_Detuning_th = H1f(nf, L - 2)
    H_int_th = H2srdef(nf, nf, L - 2)
    H_th = ( Omega/2 )*H_Rabi_th + Delta*H_Detuning_th + H_int_th
    bath = (spa.expm( - beta * H_th))/( spa.expm(-beta * H_th ).diagonal().sum().real )
    #rho_loc = 0.25*np.identity(4)
    #p={}
    #s=0.
    rho_loc=np.array([np.random.normal() for i in range(16)]).reshape(4,4)+1j*np.array([np.random.normal() for i in range(16)]).reshape(4,4)
    rho_loc=rho_loc.dot(rho_loc.conjugate().transpose())
    rho_loc=rho_loc/rho_loc.trace()
    #for mu in names[1:]:
    #    for nu in names[1:]:
    #        p[mu+nu]=np.random.uniform(0,1)
    #        s+=p[mu+nu]*p[mu+nu]
    #for key in p.keys(): p[key]=(p[key]*np.sqrt(np.random.uniform(0,1)))/np.sqrt(s)
    #for mu in names[1:]:
    #    for nu in names[1:]:
    #        rho_loc = rho_loc +  p[mu+nu]*0.25*sps.kron(pauli_dict[mu], pauli_dict[nu])
    
    rho = sps.kron(rho_loc, bath)
    rho_in  = sps.csc_matrix(rho)
    for i in names: 
       # names_pauli2.append(i+str(sites[0])+'I'+str(sites[1]))
       # paulis2.append(one_F(pauli_dict[i],sites[0],L))
        for j in names:
            names_pauli2.append(i+str(sites[0])+j+str(sites[1]))
            paulis2.append(two_FF(pauli_dict[i],pauli_dict[j],sites[0],sites[1],L))
    #O0 = one_F(Id, site, L)
    #O1 = one_F(sigma_x, site, L)
    #O2 = one_F(sigma_y, site, L)
    #O3 = one_F(sigma_z, site, L)
    for i in range(len(names_pauli2)):
        observables_dict[names_pauli2[i]]=((paulis2[i].dot(rho_in).diagonal().sum())/ 
                                           rho_in.diagonal().sum()).real
    #OId = ((O0.dot(rho_in).diagonal().sum())/ rho_in.diagonal().sum()).real 
    #Ox =  ((O1.dot(rho_in).diagonal().sum())/ rho_in.diagonal().sum()).real 
    #Oy =  ((O2.dot(rho_in).diagonal().sum())/ rho_in.diagonal().sum()).real 
    #Oz =  ((O3.dot(rho_in).diagonal().sum())/ rho_in.diagonal().sum()).real 
    #arr = [[ observables_dict[key] for key in observables_dict.keys()  ]]
    arr = [[ observables_dict[key] for key in observables_dict.keys()  ][1:]]
    print(arr)
    #print(arr)
    U = spa.expm( -1j * H * dt)
    print(U.shape)
    U_dag = spa.expm( 1j * H * dt)
    temp = [ 0.0 ]
    for i in range( 1, nstep ):
           rho_in = ((U.dot(rho_in)).dot(U_dag) ) / (rho_in.diagonal().sum())
           print("loop" )
           for k in numba.prange(len(names_pauli2)):
           #for k in range(len(names_pauli2)):

            #   print("traccia" )
           #rho_in = rho_in.toarray()
           #print(paulis2.shape)
           #exit()
           #obs = (oe.contract('kij,ji ->k',paulis2,rho_in) / rho_in.diagonal().sum()).real
               observables_dict[names_pauli2[k]]=((paulis2[k].dot(rho_in).diagonal().sum())/ rho_in.diagonal().sum()).real
           time = i * dt
           arr.append( [ observables_dict[key] for key in observables_dict.keys()  ][1:] )
           #arr.append( [ obs[key] for key in range(len(rho_in))  ][1:] )
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
        for j in range(len(cfg["system"]["beta"])):
            C_def = cfg["system"]["potential"][k]
            Delta = cfg["system"]["Delta"]
            beta = cfg["system"]["beta"][j]


            print("----Generating trajectories for impurity = %s" % C_def)


            H_Rabi = H1f(sigma_x, L)
            H_Detuning = H1f(nf, L)
            H_int = H3srdef(nf, nf, L)
            H_def = H3srdef1(nf, nf, L)
            T = cfg["time"]
            dt = cfg["dt"]
            nstep = int(T/dt)
            time = np.linspace(0 , T-2*dt , nstep - 1 , endpoint = True)

            sites = [1,2]

    
            H = ( Omega/2 )*H_Rabi + Delta*H_Detuning + H_int + C_def*H_def
    


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
                p_t1, p_t2, t, p_0, p_T = compute_traj2(sites, beta ,cfg["dt"], cfg["time"])

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

            path = os.path.join(cfg["model_dir"], f"traj_fss_dyn_{beta}_{C_def}_{L}_two_bodies_diag_gm.h5")
            path_eval_fin = os.path.join(cfg["model_dir"], f"traj_fss_final_{beta}_{C_def}_{L}_two_bodies_diag_gm.h5")

            print([v.shape for k, v in trajectories.items()])
            print(f"saving to {path}")
            dd.io.save(path, trajectories)
            dd.io.save(path_eval_fin, trajectories_eval_fin)
            print("done")
            metrics = {'dummy': 0}

