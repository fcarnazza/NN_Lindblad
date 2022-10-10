import deepdish as dp
import torch
import numpy as np




class Data_rho():

    def __init__(self, file_name):
        ds = dp.io.load(file_name)
        self.diag_rhos_t1 = ds['diag_rhos_t1']
        self.diag_rhos_t2 = ds['diag_rhos_t2']
        self.gamma = ds['gamma']
        self.inter = ds['inter']
        self.n_spins = ds['n_spins']
        self.omega = ds['omega']
        self.num = len(self.diag_rhos_t1)
        
        #this method returns a single datapoint and will later on be used to sample full batches

    def __getitem__(self, idx):
        
        v = self.diag_rhos_t1[idx].real
        v1 = self.diag_rhos_t2[idx].real
        vec = np.concatenate((v, np.array([self.gamma[idx]]), np.array([self.inter[idx]]), np.array([self.omega[idx]])))
        vec1 = np.concatenate((v1, np.array([self.gamma[idx]]), np.array([self.inter[idx]]), np.array([self.omega[idx]])))
        return torch.tensor(vec), torch.tensor(vec1)
    
    def __len__(self):
        return self.num



class Data_p_train():

    def __init__(self, file_name):
        ds = dp.io.load(file_name)
        self.p_t1 = ds['p_t1']
        self.p_t2 = ds['p_t2']
        self.num = int(0.8*len(self.p_t1))
        
        #this method returns a single datapoint and will later on be used to sample full batches

    def __getitem__(self, idx):
        
        v = self.p_t1[idx]
        v1 = self.p_t2[idx]
        return torch.tensor(v), torch.tensor(v1)
    
    def __len__(self):
        return self.num


class Data_p_eval():

    def __init__(self, file_name):
        ds = dp.io.load(file_name)
        self.p_t1 = ds['p_t1']
        self.p_t2 = ds['p_t2']
        self.num = int(0.2*len(self.p_t1))

        
        #this method returns a single datapoint and will later on be used to sample full batches

    def __getitem__(self, idx):
        
        l = len(self.p_t1) - int(0.2*len(self.p_t1))
        v = self.p_t1[l + idx]
        v1 = self.p_t2[l + idx]
        return torch.tensor(v), torch.tensor(v1)
    
    def __len__(self):
        return self.num



class Data_p_final():

    def __init__(self, file_name):
        ds = dp.io.load(file_name)
        self.p_0 = ds['p_0']
        self.p_T = ds['p_T']
        self.num = len(self.p_T)
        
        #this method returns a single datapoint and will later on be used to sample full batches

    def __getitem__(self, idx):
        
        v = self.p_0[idx]
        v1 = self.p_T[idx]
        return torch.tensor(v), torch.tensor(v1)
    
    def __len__(self):
        return self.num


class Data_p_t():

    def __init__(self, file_name):
        ds = dp.io.load(file_name)
        self.p_t1 = ds['p_t1']
        self.p_t2 = ds['p_t2']
        self.time = ds['t']
        self.num = int(len(self.p_t1)/2)
        
        #this method returns a single datapoint and will later on be used to sample full batches

    def __getitem__(self, idx):
        l = len(self.p_t1)
        v = self.p_t1[idx]
        v1 = self.p_t2[idx]
        t = [self.time[idx]]
        return torch.tensor(v), torch.tensor(v1), torch.tensor(t)
    
    def __len__(self):
        return self.num



class Data_p_t_eval():

    def __init__(self, file_name):
        ds = dp.io.load(file_name)
        self.p_t1 = ds['p_t1']
        self.p_t2 = ds['p_t2']
        self.time = ds['t']
        self.num = int(len(self.p_t1))
        
        #this method returns a single datapoint and will later on be used to sample full batches

    def __getitem__(self, idx):
        
        l = len(self.p_t1) - int(0.2*len(self.p_t1))
        v = self.p_t1[l + idx]
        v1 = self.p_t2[l + idx]
        t = [self.time[l + idx]]
        return torch.tensor(v), torch.tensor(v1), torch.tensor(t)
    
    def __len__(self):
        return self.num


class Data_p_final_t():

    def __init__(self, file_name, t_final):
        ds = dp.io.load(file_name)
        self.p_0 = ds['p_0']
        self.p_T = ds['p_T']
        self.num = len(self.p_T)
        self.t_fin = t_final
        
        #this method returns a single datapoint and will later on be used to sample full batches

    def __getitem__(self, idx):
        
        v = self.p_0[idx]
        v1 = self.p_T[idx]
        arr = np.concatenate((v,0.0), axis = None)
        arr1 =  np.concatenate((v1,self.t_fin), axis = None)
        return torch.tensor(arr), torch.tensor(arr1)
    
    def __len__(self):
        return self.num

'''class Data_p_single_traj():

    def __init__(self, file_name):
        ds = dp.io.load(file_name)
        self.p_t1 = ds['p_t1']
        self.p_t2 = ds['p_t2']
        self.num = len(self.p_t1) - 10
        
        #this method returns a single datapoint and will later on be used to sample full batches

    def __getitem__(self, idx):
        
        v = self.p_t1[idx]
        v1 = self.p_t2[idx + 9]
        return torch.tensor(v), torch.tensor(v1)
    
    def __len__(self):
        return self.num'''
class Data_p_t_cut():

    def __init__(self, file_name, T, dt):
        ds = dp.io.load(file_name)
        self.p_t1 = ds['p_t1']
        self.p_t2 = ds['p_t2']
        self.time = ds['t']
        self.l = int(T/dt)
        
        #this method returns a single datapoint and will later on be used to sample full batches

    def __getitem__(self, idx):
        l_in = int(len(self.p_t1)/100)
        temp = np.reshape(self.p_t1, (l_in, 4, 100))
        temp1 = temp[:self.l, :, :]
        v = temp1.reshape(self.l*100, 4)[idx]
        temp2 = np.reshape(self.p_t2, (l_in, 4, 100))
        temp2 = temp2[:self.l, :, :]
        v1 = temp2.reshape(self.l*100, 4)[idx]
        a = self.time.reshape(100,1999).transpose()
        b = a[:self.l, :]
        c = b.transpose()
        t = [c.reshape(b.size)[idx]]
        return torch.tensor(v), torch.tensor(v1), torch.tensor(t)
    
    def __len__(self):
        return self.l
