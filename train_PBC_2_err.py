from ml.utils import update_params_from_cmdline, save_metrics_params, ensure_dir, ensure_empty_dir, infinite_dataset
from torch.utils.data import Dataset, DataLoader
from ml.pytorch_modules.vae import BVAE
from ml.pytorch_modules.dynamics_mlp import MLP,MLLP
from torch import optim
from train_data import *
from utils.settings import cfg
import os
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import l4
import sfw
import sys
default_parameters = {'model_dir': './results/test1',
                      'n_spins': 9,
                      'batch_size': 256,
                      'batches_per_epoch': 256,
                      'n_epochs': 25,
                      #'beta': [ sub_b],
	              #'potential': [sub_c  ],
                      'beta':      [0.0,0.8,5.0,2.0,0.0,1.5,2.0,0.8,0.3,5.0,0.8,5.0,0.3,1.0,2.0 ],
                      'potential': [2.0,2.0,1.0,0.8,0.6,0.6,0.6,0.5,0.4,0.4,0.3,0.3,0.2,0.1,0.1 ],
                      'device': 'cpu',
                      'MLP_pr': {
                      		'mlp_params': {
                          		'data_dim': 15,
                          		'layers': [ ],
                              'nonlin': 'id',
                              'output_nonlin': 'id'
                          					},
                          },
                      }




def train(model, optimizer, data_loader, batches_per_epoch, epoch ):#, #writer):
    model.train()
    summed_train_loss = 0.
    for batch_idx in range(batches_per_epoch):
        constraints = sfw.constraints.create_simplex_constraints(model)
        (batch_in, batch_out) = next(data_loader)
        data_in = batch_in.float().to(cfg.device)
        data_out = batch_out.float().to(cfg.device)
        optimizer.zero_grad()
        recon_batch = model(data_in)
        loss = model.loss(data_out, recon_batch)
        summed_train_loss += loss.item()
        loss.backward()
        optimizer.step(constraints)
        #wrap.step(loss)
        #writer.add_scalar('train_loss', loss, batches_per_epoch * epoch + batch_idx)

    print('=== Mean train loss: {:.12f}'.format(summed_train_loss / batches_per_epoch))
    return model, optimizer



def eval(data_loader, model, batch_idx ):#, #writer):
    model.eval()
    summed_eval_loss = 0.
    summed_eval_test_loss = 0.
    batches_per_eval = 100
    for i in range(batches_per_eval):
        (batch_in, batch_out) = next(data_loader)
        data_in = batch_in.float().to(cfg.device)
        data_out = batch_out.float().to(cfg.device)
        recon_batch = model(data_in)
        eval_loss = model.loss(data_out, recon_batch).item()
        summed_eval_loss += eval_loss

    #writer.add_scalar('eval_loss', summed_eval_loss / batches_per_eval, batch_idx)
    print('=== Test set loss: {:.12f}'.format(summed_eval_loss / batches_per_eval))
    return {'loss': summed_eval_loss}

if __name__ == '__main__':
    cfg = update_params_from_cmdline(default_params=default_parameters)
    ensure_dir(cfg.model_dir)
    ensure_empty_dir(cfg.model_dir)
    L = cfg["n_spins"]
    for k in range(len(cfg["beta"])):
        for j in range(len(cfg["potential"])):
            beta = cfg["beta"][k]
            C = cfg["potential"][k]
            print('===Training model for beta = %s and C = %s' %(beta, C))
            #data_train = Data_p_train("./results/data_th_periodic/T_10/traj_%s_%s.h5" %(beta, C))
            data_train = Data_p_train("./results/data_th_periodic/T_10/two_bodies/traj_fss_dyn_%s_%s_%s_two_bodies_diag_gm.h5" %(beta, C,L))
            data_eval = Data_p_eval("./results/data_th_periodic/T_10/two_bodies/traj_fss_dyn_%s_%s_%s_two_bodies_diag_gm.h5" %(beta, C,L))
            train_loader = DataLoader(dataset=data_train, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
            train_loader = infinite_dataset(train_loader)
            eval_loader = DataLoader(dataset=data_eval, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
            eval_loader = infinite_dataset(eval_loader)
        #    #writer = SummaryWriter(os.path.join(cfg.model_dir, 'tensorboard'))
            model = MLLP(**cfg.MLP_pr).to(cfg.device)
            optimizer = sfw.optimizers.Adam(model.parameters() ,lr =0.01)#optim.Adam(model.parameters(), lr=0.01)
            #wrap = l4.L4(optimizer )
            for epoch in range(cfg.n_epochs):
                 print('= Starting epoch ', epoch, '/', cfg.n_epochs)
                 model, optimizer = train(model, optimizer, train_loader, cfg.batches_per_epoch, epoch )#, #writer)
                 metrics = eval(eval_loader, model, cfg.batches_per_epoch * epoch )#, #writer)
 
            save_metrics_params(metrics, cfg)


            torch.save(model.state_dict(), './trained_models/PBC/Thermal/eff__Cdef_%s_beta_%s_L_%s_two_bodies_LL' %(C, beta,L))
