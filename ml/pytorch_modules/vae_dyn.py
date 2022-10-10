import torch
from ml.pytorch_modules.architectures import *

encoder_dict = {'mlp': MLPEncoder}

decoder_dict = {'mlp': MLPDecoder}


class BVAE_dyn(nn.Module):
    

    def __init__(self, beta, encoder_str, decoder_str, latent_size, encoder_params, decoder_params, mlp_params,
                 rec_loss_fn=torch.nn.MSELoss()):
        super().__init__()
        self.beta = beta
        self.rec_loss_fn = rec_loss_fn
        self.encoder = encoder_dict[encoder_str](latent_size=latent_size, **encoder_params)
        self.decoder = decoder_dict[decoder_str](latent_size=latent_size, **decoder_params)
        self.dynamics = MLP_simple(**mlp_params)
    

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z_t1 = self.reparameterize(mu, logvar)
        z_t2 = self.dynamics(z_t1)
        return self.decoder(z_t1) , mu, logvar, self.decoder(z_t2)
    

    def loss(self, xt1, recon_xt1, xt2, recon_xt2, mu, logvar):
        rec_t1 = self.rec_loss_fn(recon_xt1, xt1)
        rec_t2 = self.rec_loss_fn(recon_xt2, xt2)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return rec_t1 + self.beta * kl + rec_t2