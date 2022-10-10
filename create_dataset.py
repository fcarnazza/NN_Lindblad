import scipy.sparse.linalg as spa
import scipy.sparse as sps
import os
import time
from physics.FSS_dyn.ED import *
import deepdish as dd

from ml.utils import on_cluster

if on_cluster():
    from cluster import update_params_from_cmdline, save_metrics_params
else:
    from ml.utils import update_params_from_cmdline, save_metrics_params
from ml.utils import ensure_dir, ensure_empty_dir, infinite_dataset, recursive_objectify, update_recursive

default_parameters = {'working_dir': './results/ds/T_10',
                      'n_spins': 5,  # system size (choose odd for OBC!)
                      'alpha_or_beta': 1.2,
                      'potential': 0.1,
                      'rabi': 1,
                      'delta': 0,
                      'dt': 0.01,  # time step
                      'total_time': 10,  # total time of the evolution
                      'hold_out_intervals': [],
                      'num_trajectories': 10,  # number
                      'name': 'train',
                      'seed': 0,
                      'system': 'obc',
                      'sequential': True
                      }


def compute_trajectory(n_spins, rabi, potential, alpha_or_beta, delta, total_time, dt, hold_out_intervals, rs, system):
    sigma_x = np.array([[0., 1.], [1., 0.]])
    N = np.array([[0., 0.], [0., 1.]])
    sigma_z = np.array([[1., 0.], [0., -1.]])
    sigma_y = np.array([[0., -1j], [1j, 0]])
    Pj = np.array([[1., 0.], [0., 0.]])
    nf = np.array([[1., 0], [0, 0]])  ###  local fermions number nf = 1-sz ###
    Id = np.eye(2).astype(np.float64)

    L = n_spins
    Omega = rabi
    C = potential
    T = total_time
    dt = dt

    def compute_traj_OBC(dt, T, hold_out_intervals):
        H_Rabi = H1f(sigma_x, L)
        H_Detuning = H1f(nf, L)

        H_lr_ferm = H2fflr(nf, nf, L, alpha_or_beta)
        site = (L + 1) / 2
        H = (Omega / 2) * H_Rabi + delta * H_Detuning + C * H_lr_ferm
        nstep = int(T / dt)
        # arr1 = np.random.rand(2)
        a = rs.uniform(0, 1)
        b = rs.uniform(0, 1)
        c = rs.uniform(0, 1)
        x = np.sqrt(a / 3)
        y = np.sqrt(b / 3)
        z = np.sqrt(c / 3)
        rho_loc = 0.5 * np.array([[1 - z, x - 1j * y], [x + 1j * y, 1 + z]])
        red = sps.csc_matrix(rho_loc)
        bath = sps.eye(2 ** ((L - 1) / 2))
        rho_in = sps.kron(bath, sps.kron(red, bath))
        O0 = one_F(Id, site, L)
        O1 = one_F(sigma_x, site, L)
        O2 = one_F(sigma_y, site, L)
        O3 = one_F(sigma_z, site, L)
        OId = ((O0.dot(rho_in).diagonal().sum()) / rho_in.diagonal().sum()).real
        Ox = ((O1.dot(rho_in).diagonal().sum()) / rho_in.diagonal().sum()).real
        Oy = ((O2.dot(rho_in).diagonal().sum()) / rho_in.diagonal().sum()).real
        Oz = ((O3.dot(rho_in).diagonal().sum()) / rho_in.diagonal().sum()).real
        arr = [[OId, Ox, Oy, Oz]]
        U = spa.expm(-1j * H * dt)
        U_dag = spa.expm(1j * H * dt)
        temp = [0.0]
        for i in range(1, nstep):
            t_start = time.time()
            rho_in = ((U.dot(rho_in)).dot(U_dag)) / (rho_in.diagonal().sum())
            OId = ((O0.dot(rho_in).diagonal().sum()) / rho_in.diagonal().sum()).real
            Ox = ((O1.dot(rho_in).diagonal().sum()) / rho_in.diagonal().sum()).real
            Oy = ((O2.dot(rho_in).diagonal().sum()) / rho_in.diagonal().sum()).real
            Oz = ((O3.dot(rho_in).diagonal().sum()) / rho_in.diagonal().sum()).real
            t = i * dt
            add_timestep = True
            for interval in hold_out_intervals:
                if t > interval[0] and t < interval[1]:
                    add_timestep = False
            if add_timestep:
                arr.append([OId, Ox, Oy, Oz])
                temp.append(t)
            # print(time.time()-t_start)
        print("finished Time = %s" % T)
        return np.asarray(arr[:-1]).astype(np.float32), \
               np.asarray(arr[1:]).astype(np.float32), \
               np.asarray(temp[:-1]).astype(np.float32), \
               np.expand_dims(np.asarray(arr[0]).astype(np.float32), 0), \
               np.expand_dims(np.asarray(arr[-1]).astype(np.float32), 0)

    def compute_traj_PBC(dt, T, hold_out_intervals):
        H_Rabi = H1f(sigma_x, L)
        H_Detuning = H1f(nf, L)

        H_sr = H2srdef(nf, nf, L)
        H_sr1 = H1srdef(nf, nf, L)
        site = 1
        H = (Omega / 2) * H_Rabi + delta * H_Detuning + H_sr + C * H_sr1

        nstep = int(T / dt)
        H_Rabi_th = H1f(sigma_x, L - 1)
        H_Det_th = H1f(nf, L - 1)
        H_sr_th = H2sr(nf, nf, L - 1)
        H_th = (Omega / 2) * H_Rabi_th + delta * H_Det_th + H_sr_th
        bath = (spa.expm(- alpha_or_beta * H_th)) / (spa.expm(-alpha_or_beta * H_th).diagonal().sum().real)
        a = rs.uniform(0, 1)
        b = rs.uniform(0, 1)
        c = rs.uniform(0, 1)
        x = np.sqrt(a / 3)
        y = np.sqrt(b / 3)
        z = np.sqrt(c / 3)
        rho_loc = 0.5 * np.array([[1 - z, x - 1j * y], [x + 1j * y, 1 + z]])
        rho = sps.kron(rho_loc, bath)
        rho_in = sps.csc_matrix(rho)
        O0 = one_F(Id, site, L)
        O1 = one_F(sigma_x, site, L)
        O2 = one_F(sigma_y, site, L)
        O3 = one_F(sigma_z, site, L)
        OId = ((O0.dot(rho_in).diagonal().sum()) / rho_in.diagonal().sum()).real
        Ox = ((O1.dot(rho_in).diagonal().sum()) / rho_in.diagonal().sum()).real
        Oy = ((O2.dot(rho_in).diagonal().sum()) / rho_in.diagonal().sum()).real
        Oz = ((O3.dot(rho_in).diagonal().sum()) / rho_in.diagonal().sum()).real
        arr = [[OId, Ox, Oy, Oz]]
        U = spa.expm(-1j * H * dt)
        U_dag = spa.expm(1j * H * dt)
        temp = [0.0]
        for i in range(1, nstep):
            t_start = time.time()
            rho_in = ((U.dot(rho_in)).dot(U_dag)) / (rho_in.diagonal().sum())
            OId = ((O0.dot(rho_in).diagonal().sum()) / rho_in.diagonal().sum()).real
            Ox = ((O1.dot(rho_in).diagonal().sum()) / rho_in.diagonal().sum()).real
            Oy = ((O2.dot(rho_in).diagonal().sum()) / rho_in.diagonal().sum()).real
            Oz = ((O3.dot(rho_in).diagonal().sum()) / rho_in.diagonal().sum()).real
            t = i * dt
            add_timestep = True
            for interval in hold_out_intervals:
                if t > interval[0] and t < interval[1]:
                    add_timestep = False
            if add_timestep:
                arr.append([OId, Ox, Oy, Oz])
                temp.append(t)
            # print(time.time()-t_start)
        print("finished Time = %s" % T)
        return np.asarray(arr[:-1]).astype(np.float32), \
               np.asarray(arr[1:]).astype(np.float32), \
               np.asarray(temp[:-1]).astype(np.float32), \
               np.expand_dims(np.asarray(arr[0]).astype(np.float32), 0), \
               np.expand_dims(np.asarray(arr[-1]).astype(np.float32), 0)

    def compute_traj_hyper_eval(dt, T, hold_out_intervals):
        H_Rabi = H1f(sigma_x, L)
        H_Detuning = H1f(nf, L)
        H_sr = H2srdef(nf, nf, L)
        H_sr1 = H1srdef(nf, nf, L)
        site = 1
        H = (Omega / 2) * H_Rabi + delta * H_Detuning + H_sr + C * H_sr1
        nstep = int(T / dt)
        H_Rabi_th = H1f(sigma_x, L - 1)
        H_Det_th = H1f(nf, L - 1)
        H_sr_th = H2sr(nf, nf, L - 1)
        H_th = (Omega / 2) * H_Rabi_th + delta * H_Det_th + H_sr_th
        bath = (spa.expm(- alpha_or_beta * H_th)) / (spa.expm(-alpha_or_beta * H_th).diagonal().sum().real)
        a = rs.uniform(0.1, 0.7)
        rho_loc = np.array([[a, 0], [0, 1 - a]])
        rho = sps.kron(rho_loc, bath)
        rho_in = sps.csc_matrix(rho)
        O0 = one_F(Id, site, L)
        O1 = one_F(sigma_x, site, L)
        O2 = one_F(sigma_y, site, L)
        O3 = one_F(sigma_z, site, L)
        OId = ((O0.dot(rho_in).diagonal().sum()) / rho_in.diagonal().sum()).real
        Ox = ((O1.dot(rho_in).diagonal().sum()) / rho_in.diagonal().sum()).real
        Oy = ((O2.dot(rho_in).diagonal().sum()) / rho_in.diagonal().sum()).real
        Oz = ((O3.dot(rho_in).diagonal().sum()) / rho_in.diagonal().sum()).real
        arr = [[OId, Ox, Oy, Oz]]
        U = spa.expm(-1j * H * dt)
        U_dag = spa.expm(1j * H * dt)
        temp = [0.0]
        for i in range(1, nstep):
            t_start = time.time()
            rho_in = ((U.dot(rho_in)).dot(U_dag)) / (rho_in.diagonal().sum())
            OId = ((O0.dot(rho_in).diagonal().sum()) / rho_in.diagonal().sum()).real
            Ox = ((O1.dot(rho_in).diagonal().sum()) / rho_in.diagonal().sum()).real
            Oy = ((O2.dot(rho_in).diagonal().sum()) / rho_in.diagonal().sum()).real
            Oz = ((O3.dot(rho_in).diagonal().sum()) / rho_in.diagonal().sum()).real
            t = i * dt
            add_timestep = True
            for interval in hold_out_intervals:
                if t > interval[0] and t < interval[1]:
                    add_timestep = False
            if add_timestep:
                arr.append([OId, Ox, Oy, Oz])
                temp.append(t)
            # print(time.time()-t_start)

        print("finished Time = %s" % T)
        return np.asarray(arr[:-1]).astype(np.float32), \
               np.asarray(arr[1:]).astype(np.float32), \
               np.asarray(temp[:-1]).astype(np.float32), \
               np.expand_dims(np.asarray(arr[0]).astype(np.float32), 0), \
               np.expand_dims(np.asarray(arr[-1]).astype(np.float32), 0)

    if system == "obc":
        compute_traj = compute_traj_OBC
    elif system == "pbc":
        compute_traj = compute_traj_PBC
    elif system == "hyper_eval":
        compute_traj = compute_traj_hyper_eval
    else:
        raise ValueError(system)
    print(f"----Generating trajectories for {system}, {C}, {alpha_or_beta}")
    return compute_traj(dt, T, hold_out_intervals)


def create_ds(working_dir, n_spins, rabi, potential, alpha_or_beta, delta, total_time, dt, num_trajectories, name,
              hold_out_intervals, seed, system, sequential):
    ensure_dir(working_dir)
    rs = np.random.RandomState(seed)

    trajectories_lst = [
        compute_trajectory(n_spins, rabi, potential, alpha_or_beta, delta, total_time, dt, hold_out_intervals, rs,
                           system)
        for _
        in range(num_trajectories)]

    path_seq = ""
    if sequential:
        trajectories_seq = {'p_ts': [(p_t1, p_t2) for p_t1, p_t2, _, _, _ in trajectories_lst]}
        path_seq = os.path.join(working_dir, f"traj_fss_sequential_{system}_{alpha_or_beta}_{potential}_{name}.h5")
        dd.io.save(path_seq, trajectories_seq)

    p_t1s, p_t2s, ts, p_0s, p_Ts = zip(*trajectories_lst)
    p_t1s, p_t2s, ts, p_0s, p_Ts = np.concatenate(p_t1s), np.concatenate(p_t2s), np.concatenate(ts), np.concatenate(
        p_0s), np.concatenate(p_Ts)

    # p_t1s, p_t2s, ts, p_0s, p_Ts = compute_trajectories(n_spins, rabi, potential, alpha_or_beta, delta, total_time, dt,
    #                                                    num_trajectories, rs, system)
    trajectories = {'p_t1': p_t1s, 'p_t2': p_t2s, 't': ts}
    trajectories_eval_fin = {'p_0': p_0s, 'p_T': p_Ts}

    path = os.path.join(working_dir, f"traj_fss_dyn_{system}_{alpha_or_beta}_{potential}_{name}.h5")
    path_eval_fin = os.path.join(working_dir, f"traj_fss_final_{system}_{alpha_or_beta}_{potential}_{name}.h5")

    dd.io.save(path, trajectories)
    dd.io.save(path_eval_fin, trajectories_eval_fin)
    return path, path_eval_fin, path_seq

def run_grid_local(cfg):
    for C in cfg["system"]["potential"]:
        for alpha_or_beta in cfg["system"]["alpha_or_beta"]:
            create_ds(cfg["working_dir"],
                      cfg["seed"],
                      cfg["system"]["n_spins"],
                      cfg["system"]["Rabi"],
                      C,
                      alpha_or_beta,
                      cfg["system"]["Delta"],
                      cfg["total_time"],
                      cfg["dt"],
                      cfg["num_trajectories"])


if __name__ == '__main__':
    if on_cluster():
        cfg_cmd = update_params_from_cmdline()
        cfg = update_recursive(default_parameters, cfg_cmd)
        cfg = recursive_objectify(cfg)
    else:
        cfg = update_params_from_cmdline(default_params=default_parameters)
    create_ds(**cfg)
    save_metrics_params({}, cfg)
