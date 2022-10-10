import torch
from ml.pytorch_modules.architectures import *

encoder_dict = {'mlp': MLPEncoder}

decoder_dict = {'mlp': MLPDecoder}



class BVAE(nn.Module):
    def __init__(self, beta, encoder_str, decoder_str, latent_size, encoder_params, decoder_params,
                 rec_loss_fn=torch.nn.MSELoss()):
        super().__init__()
        self.beta = beta
        self.rec_loss_fn = rec_loss_fn
        self.encoder = encoder_dict[encoder_str](latent_size=latent_size, **encoder_params)
        self.decoder = decoder_dict[decoder_str](latent_size=latent_size, **decoder_params)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss(self, x, recon_x, mu, logvar):
        rec = self.rec_loss_fn(recon_x, x)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return rec + self.beta * kl



