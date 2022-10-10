import scipy.sparse.linalg as spa
import scipy.sparse as sps
from ml.utils import update_params_from_cmdline, save_metrics_params, ensure_dir
import os
from physics.FSS_dyn.ED_b import *
import deepdish as dd


default_parameters = {'model_dir': './results/data_th_periodic/T_10',
                      'system': {
                          'n_spins': 7,  # system size
                           #'alpha' : [ 6, 2, 0 ],
                           #'beta': [0, 0.2, 0.3, 0.4, 0.6, 1, 5],
                           #'potential': [0.2, 0.4, 0.5, 0.6, 0.8, 1, 1.1],
                           'beta':[2]#,3,4],
                           'potential':[0.1]#, 0.3, 0.5, 0.7, 0.8],
                           'Rabi': 1,
                           'Delta': 0,

                      },
                      'dt': 0.01,  # time step
                      'time': 10,  # total time of the evolution
                      'num_trajectories': 100,  # number
                      'seed': None
                      }

#	**********  BASIC MATRIX DEFINITIONS  **********
#
sigma_x = np.array( [ [ 0. , 1. ] , [ 1. , 0. ] ] )
N = np.array( [ [ 0. , 0. ] , [ 0. , 1. ] ] )
sigma_z = np.array( [ [ 1. , 0. ] , [ 0. , -1. ] ] )
sigma_y = np.array([[0, -1j], [1j, 0]])
Pj = np.array( [ [ 1. , 0. ] , [ 0. , 0. ] ] )
nf = np.array([[1, 0],[0, 0]]) ###  local fermions number nf = 1-sz ###
Id = np.eye(2)
pauli_dict = {
    'X': sigma_x,
    'Y': sigma_y,
    'Z': sigma_z,
    'I': Id
}
names=['I','X','Y','Z']


def compute_traj(site, beta, dt, T):
    nstep = int(T / dt)
    H_Rabi_th = H1f(sigma_x, L - 1)
    H_Detuning_th = H1f(nf, L - 1)
    H_int_th = H2srdef(nf, nf, L - 1)
    H_th = ( Omega/2 )*H_Rabi_th + Delta*H_Detuning_th + H_int_th
    bath = (spa.expm( - beta * H_th))/( spa.expm(-beta * H_th ).diagonal().sum().real )
    a = np.random.uniform(0,1)
    b = np.random.uniform(0,1)
    c = np.random.uniform(0,1)
    x = np.sqrt(a/3)
    y = np.sqrt(b/3)
    z = np.sqrt(c/3)
    rho_loc = 0.5*np.array( [ [ 1 - z , x - 1j*y ],  [  x +1j*y , 1 + z ] ] )
    rho = sps.kron(rho_loc, bath)
    rho_in  = sps.csc_matrix(rho)
    O0 = one_F(Id, site, L)
    O1 = one_F(sigma_x, site, L)
    O2 = one_F(sigma_y, site, L)
    O3 = one_F(sigma_z, site, L)
    OId =  ((O0.dot(rho_in).diagonal().sum())/ rho_in.diagonal().sum()).real
    Ox =  ((O1.dot(rho_in).diagonal().sum())/ rho_in.diagonal().sum()).real 
    Oy =  ((O2.dot(rho_in).diagonal().sum())/ rho_in.diagonal().sum()).real 
    Oz =  ((O3.dot(rho_in).diagonal().sum())/ rho_in.diagonal().sum()).real 
    arr = [ [ OId, Ox, Oy, Oz ]  ]
    print(arr)
    U = spa.expm( -1j * H * dt)
    U_dag = spa.expm( 1j * H * dt)
    temp = [ 0.0 ]
    for i in range( 1, nstep ):
        rho_in = ((U.dot(rho_in)).dot(U_dag) ) / (rho_in.diagonal().sum())
        OId =  ((O0.dot(rho_in).diagonal().sum())/ rho_in.diagonal().sum()).real
        Ox =  ((O1.dot(rho_in).diagonal().sum())/ rho_in.diagonal().sum()).real 
        Oy =  ((O2.dot(rho_in).diagonal().sum())/ rho_in.diagonal().sum()).real 
        Oz =  ((O3.dot(rho_in).diagonal().sum())/ rho_in.diagonal().sum()).real
        time = i * dt
        #if time in times:  
        arr.append( [ OId, Ox, Oy, Oz ] )
        temp.append( i * dt )
    print("finished Time = %s" % T)
    return np.asarray(arr[:-1]).astype(np.float32), \
           np.asarray(arr[1:]).astype(np.float32), \
           np.asarray(temp[:-1]).astype(np.float32), \
           np.expand_dims(np.asarray(arr[0]).astype(np.float32), 0), \
           np.expand_dims(np.asarray(arr[-1]).astype(np.float32), 0)

def compute_traji2(site, beta, dt, T):
    # Creating a dicrionary for the two spin Paulis
    names_pauli2=[]
    paulis2=[]
    observables_dict={}
    nstep = int(T / dt)
    H_Rabi_th = H1f(sigma_x, L - 2)
    H_Detuning_th = H1f(nf, L - 2)
    H_int_th = H2srdef(nf, nf, L - 2)
    H_th = ( Omega/2 )*H_Rabi_th + Delta*H_Detuning_th + H_int_th
    bath = (spa.expm( - beta * H_th))/( spa.expm(-beta * H_th ).diagonal().sum().real )
    # Creating a random matrix with the characteristcs of a density matrix
    rho_loc=np.array([np.random.normal() for i in range(16)]).reshape(4,4)+1j*np.array([np.random.normal() for i in range(16)]).reshape(4,4)
    rho_loc=rho_loc.dot(rho_loc.conjugate().transpose())
    rho_loc=rho_loc/rho_loc.trace()    
    rho = sps.kron(rho_loc, bath)
    rho_in  = sps.csc_matrix(rho)
    #Creating the names X1X2, X1Y1,....
    for i in names: 
        for j in names:
            names_pauli2.append(i+str(sites[0])+j+str(sites[1]))
            paulis2.append(two_FF(pauli_dict[i],pauli_dict[j],sites[0],sites[1],L))
    #Initialiing the values in the observable dictionaty
    for i in range(len(names_pauli2)):
        observables_dict[names_pauli2[i]]=((paulis2[i].dot(rho_in).diagonal().sum())/ 
                                           rho_in.diagonal().sum()).real
    arr = [[ observables_dict[key] for key in observables_dict.keys()  ]]
    print(arr)
    U = spa.expm( -1j * H * dt)
    U_dag = spa.expm( 1j * H * dt)
    temp = [ 0.0 ]
    for i in range( 1, nstep ):
        rho_in = ((U.dot(rho_in)).dot(U_dag) ) / (rho_in.diagonal().sum())
        for k in range(len(names_pauli2)):
            observables_dict[names_pauli2[k]]=((paulis2[k].dot(rho_in).diagonal().sum())/ rho_in.diagonal().sum()).real
        time = i * dt
        #if time in times:
        arr.append( [ observables_dict[key] for key in observables_dict.keys()  ] )
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


            print("----Generating trajectories for impurity = %s and for beta = %s" % (C_def, beta))


            H_Rabi = H1f(sigma_x, L)
            H_Detuning = H1f(nf, L)
            #H_lr_ferm = H2fflr(nf, nf, L, a )
            H_sr = H2srdef(nf, nf, L)
            H_sr1 = H1srdef(nf, nf, L)
            T = cfg["time"]
            dt = cfg["dt"]
            nstep = int(T/dt)

            site = 1
           
            #times = np.random.choice(time, 900, replace = False) 
    
            H = ( Omega/2 )*H_Rabi + Delta*H_Detuning + H_sr + C_def*H_sr1 
    


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
                p_t1, p_t2, t, p_0, p_T = compute_traj(site, beta ,cfg["dt"], cfg["time"])

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

            path = os.path.join(cfg["model_dir"], f"traj_fss_dyn_{beta}_{C_def}_.h5")
            path_eval_fin = os.path.join(cfg["model_dir"], f"traj_fss_final_{beta}_{C_def}_.h5")

            print([v.shape for k, v in trajectories.items()])
            print(f"saving to {path}")
            dd.io.save(path, trajectories)
            dd.io.save(path_eval_fin, trajectories_eval_fin)
            print("done")
            metrics = {'dummy': 0}
