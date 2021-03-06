import torch
import torch.nn as nn
import torch.nn.functional as F


# define a simple linear VAE
class LinearVAE(nn.Module):
    def __init__(self, features):
        super(LinearVAE, self).__init__()

        self.features = features 

        # encoder
        self.enc1 = nn.Linear(in_features=784, out_features=512)
        self.enc2 = nn.Linear(in_features=512, out_features=features*2)

        # decoder 
        self.dec1 = nn.Linear(in_features=features, out_features=512)
        self.dec2 = nn.Linear(in_features=512, out_features=784)


    def encode(self, x): 
        # encoding
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, self.features)

        # get `mu` and `log_var` from the input space
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance

        return mu, log_var



    def decode(self, x): 
        # decoding
        x = F.relu(self.dec1(x))
        reconstruction = torch.sigmoid(self.dec2(x))
        return reconstruction



    def reparameterize(self, mu, log_var):
        """
        Reparametrisation trick: pretends sample coming from input space instead of latent vector space
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        epsilon = torch.randn_like(std) # `randn_like`, random numbers from normal distribution, as we need the same size
        sample = mu + (epsilon * std) # sampling as if coming from the input space
        return sample


    def forward(self, x):
    
        # encoding 
        mu, log_var = self.encode(x)

        # get the latent vector through reparameterization trick
        z = self.reparameterize(mu, log_var)

        #decoding 
        return self.decode(z), mu, log_var