import torch
from torch.utils.data import DataLoader

from modules_ import *

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.inconv = InConv(64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)

        self.down_mu = Down(256, 512)
        self.down_sigma = Down(256, 512)
    
    def forward(self, x):
        x = self.inconv(x)
        x = self.down1(x)
        x = self.down2(x)

        mu = self.down_mu(x)
        sigma = self.down_sigma(x)

        return mu, sigma
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)

        self.outconv = OutConv(64, 3)
    
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        x = self.outconv(x)

        return x

if __name__ == '__main__':
    encoder = Encoder()
    decoder = Decoder()

    x = torch.randn(1, 3, 256, 256)

    mu, sigma = encoder(x)
    sigma = torch.exp(sigma * 0.5)
    epsilon = torch.randn_like(sigma)
    
    z = mu + sigma * epsilon

    x = decoder(z)


    print(x.shape)
