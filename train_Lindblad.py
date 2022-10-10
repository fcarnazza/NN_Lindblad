from ml.utils import update_params_from_cmdline, save_metrics_params, ensure_dir, ensure_empty_dir, infinite_dataset
from torch.utils.data import Dataset, DataLoader
from ml.pytorch_modules.vae import BVAE
from ml.pytorch_modules.dynamics_mlp import MLLP, MLP
from torch import optim
from train_data import *
from utils.settings import cfg
import os
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./sfw')
#import optimizers as sfw_opt
#import constraints as sfw_cons
from s_const import *
default_parameters = {'model_dir': './results/test1',
                      'n_spins': 7,
                      'batch_size': 256,
                      'batches_per_epoch': 256,
                      'n_epochs': 40,
                      'potential': [0.5],
                      'gamma': [0.5],
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


def train(model, optimizer, data_loader, batches_per_epoch, epoch, writer):
    model.train()
    summed_train_loss = 0.
    #constraints = sfw_cons.create_lp_constraints(model) 
    for batch_idx in range(batches_per_epoch):
        (batch_in, batch_out) = next(data_loader)
        data_in = batch_in.float().to(cfg.device)
        data_out = batch_out.float().to(cfg.device)
        optimizer.zero_grad()
        recon_batch = model(data_in)
        loss = model.loss(data_out, recon_batch)
        summed_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        writer.add_scalar('train_loss', loss, batches_per_epoch * epoch + batch_idx)

    print('=== Mean train loss: {:.12f}'.format(summed_train_loss / batches_per_epoch))
    return model, optimizer


def eval(data_loader, model, batch_idx, writer):
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

    writer.add_scalar('eval_loss', summed_eval_loss / batches_per_eval, batch_idx)
    print('=== Test set loss: {:.12f}'.format(summed_eval_loss / batches_per_eval))
    return {'loss': summed_eval_loss}

if __name__ == '__main__':
    cfg = update_params_from_cmdline(default_params=default_parameters)
    ensure_dir(cfg.model_dir)
    ensure_empty_dir(cfg.model_dir)
    for j in range(len(cfg["potential"])):
            gamma = cfg["gamma"][j]
            print('===Training model and gamma = %s' %(gamma))
            #data_train = Data_p_train("./results/data_th_periodic/T_10/traj_%s_%s.h5" %(beta, C))
            data_train = Data_p_train("./results/data_th_periodic/T_10/two_bodies/traj_fss_dyn_%s_Lindblad_diag.h5" %( gamma))
            data_eval = Data_p_eval("./results/data_th_periodic/T_10/two_bodies/traj_fss_dyn_%s_Lindblad_diag.h5" %( gamma))
            train_loader = DataLoader(dataset=data_train, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
            train_loader = infinite_dataset(train_loader)
            eval_loader = DataLoader(dataset=data_eval, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
            eval_loader = infinite_dataset(eval_loader)
            writer = SummaryWriter(os.path.join(cfg.model_dir, 'tensorboard'))
            model = MLLP(**cfg.MLP_pr).to(cfg.device)
            optimizer = optim.Adam(model.parameters())
            for epoch in range(cfg.n_epochs):
                print('= Starting epoch ', epoch, '/', cfg.n_epochs)
                model, optimizer = train(model, optimizer, train_loader, cfg.batches_per_epoch, epoch, writer)
                metrics = eval(eval_loader, model, cfg.batches_per_epoch * epoch, writer)
            save_metrics_params(metrics, cfg)
            torch.save(model.state_dict(), './trained_models/PBC/Thermal/eff_gamma_%s_Lindblad' %(gamma))

