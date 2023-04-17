import torch
from torch.utils.data import DataLoader

from modules_ import *
from glob import glob


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.inconv = InConv(64)
        self.down1 = Down(64, 128)

        self.down_mu = Down(128, 256)
        self.down_sigma = Down(128, 256)
    
    def forward(self, x):
        x = self.inconv(x)
        x = self.down1(x)
        mu = self.down_mu(x)
        sigma = self.down_sigma(x)

        return mu, sigma

if __name__ == '__main__':
    encoder = Encoder()

    x = torch.randn(1, 1, 28, 28)

    mu, sigma = encoder(x)
    print(mu.shape, sigma.shape)
    
    sigma = torch.exp(sigma * 0.5)
    epsilon = torch.randn_like(sigma)
    
    z = mu + sigma * epsilon

    print(z.shape)
