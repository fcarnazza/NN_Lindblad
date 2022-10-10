from ml.utils import update_params_from_cmdline, save_metrics_params, ensure_dir, ensure_empty_dir, infinite_dataset
from torch.utils.data import Dataset, DataLoader
from ml.pytorch_modules.vae import BVAE
from ml.pytorch_modules.dynamics_mlp import hyper_mlp
from torch import optim
from train_data import *
from utils.settings import cfg
import os
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

default_parameters = {'model_dir': './results/test2',
                      'n_spins': 5,
                      'batch_size': 48,
                      'batches_per_epoch': 48,
                      'beta': [0.3],
                      'potential' : [0.3],
                      'n_epochs': 200  ,
                      'device': 'cpu',
                      'MLP_par': {
                      		'mlp_params': {
                              'data_dim': 16,
                              'layers': [ 50, 100, 50],
                          		'nonlin': 'tanh',
                          		'output_nonlin': 'id'
                          					},
                          },
                      }




def train(model, optimizer, data_loader, batches_per_epoch, epoch, writer):
    model.train()
    summed_train_loss = 0.
    for batch_idx in range(batches_per_epoch):
        (rho_in, rho_out, time_in) = next(data_loader)
        data_in = time_in.float().to(cfg.device)
        data_out = rho_out.float().to(cfg.device)
        data_mid = rho_in.float().to(cfg.device)
        optimizer.zero_grad()
        recon_batch, M = model(data_in, rho_in)
        loss = model.loss(data_out, recon_batch)
        summed_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        writer.add_scalar('train_loss', loss, batches_per_epoch * epoch + batch_idx)

    print('=== Mean train loss: {:.4f}'.format(summed_train_loss / batches_per_epoch))
    return model, optimizer


def eval(data_loader, model, batch_idx, writer):
    model.eval()
    summed_eval_loss = 0.
    summed_eval_test_loss = 0.
    batches_per_eval = 100
    for i in range(batches_per_eval):
        (rho_in, rho_out, time_in) = next(data_loader)
        data_in = time_in.float().to(cfg.device)
        data_out = rho_out.float().to(cfg.device)
        data_mid = rho_in.float().to(cfg.device)
        recon_batch, M = model(data_in, rho_in)
        eval_loss = model.loss(data_out, recon_batch).item()
        summed_eval_loss += eval_loss

    writer.add_scalar('eval_loss', summed_eval_loss / batches_per_eval, batch_idx)
    print('=== Test set loss: {:.4f}'.format(summed_eval_loss / batches_per_eval))
    return {'loss': summed_eval_loss}

if __name__ == '__main__':
    cfg = update_params_from_cmdline(default_params=default_parameters)
    ensure_dir(cfg.model_dir)
    ensure_empty_dir(cfg.model_dir)
    for k in range(len(cfg["beta"])):
        for j in range(len(cfg["potential"])):
            beta = cfg["beta"][k]
            C = cfg["potential"][j]
            print('===Training model for beta = %s and C = %s' %(beta, C))
            data_train = Data_p_t('./results/data_th_periodic/T_10/traj_fss_dyn_%s_%s_2_spin.h5' %(beta, C))
            data_eval = Data_p_t('./results/data_th_periodic/T_10/traj_fss_dyn_%s_%s_eval_2_spin.h5' %(beta, C))
            train_loader = DataLoader(dataset=data_train, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
            train_loader = infinite_dataset(train_loader)
            eval_loader = DataLoader(dataset=data_eval, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
            eval_loader = infinite_dataset(eval_loader)
            writer = SummaryWriter(os.path.join(cfg.model_dir, 'tensorboard'))
            model = hyper_mlp(**cfg.MLP_par).to(cfg.device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            for epoch in range(cfg.n_epochs):
                print('= Starting epoch ', epoch, '/', cfg.n_epochs)
                model, optimizer = train(model, optimizer, train_loader, cfg.batches_per_epoch, epoch, writer)
                metrics = eval(eval_loader, model, cfg.batches_per_epoch * epoch, writer)
            save_metrics_params(metrics, cfg)

            torch.save(model.state_dict(), './trained_models/PBC/hyper/eff_Cdef_%s_beta_%s_hyper_2_spin' %(C, beta))
