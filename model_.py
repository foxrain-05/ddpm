import torch

from data import DataSet
from torch.utils.data import DataLoader
from modules import *


class DiffusionModel(nn.Module):
    def __init__(self, t_range=1000, beta_small=1e-4, beta_large=0.02):
        super().__init__()

        self.t_range = t_range
        self.beta_small = beta_small
        self.beta_large = beta_large

        self.betas = torch.linspace(beta_small, beta_large, t_range)
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

        self.inc = DoubleConv(3, 64)
        self.outc = OutConv(64, 3)

        self.donw1 = Down(64, 128)
        self.donw2 = Down(128, 256)
        self.donw3 = Down(256, 512)

        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)

        self.sa1 = SAWrapper(256, 8)
        self.sa2 = SAWrapper(512, 4)

        self.pe1 = PositionalEncoding(128)
        self.pe2 = PositionalEncoding(256)
        self.pe3 = PositionalEncoding(512)
    
    def forward(self, x, t):
        x1 = self.inc(x)
        x2 = self.donw1(x1) + self.pe1(t)
        x3 = self.donw2(x2) + self.pe2(t)
        x3 = self.sa1(x3)
        x4 = self.donw3(x3) + self.pe3(t)
        x4 = self.sa2(x4)
        return x4

    def loss_fn(self, x_0):
        ts = torch.randint(0, self.t_range, size=(x_0.shape[0],), dtype=torch.int64)
        eposilon = torch.randn_like(x_0, dtype=torch.float32)
        
        alpha_t_bar = torch.gather(self.alphas_bar, 0, ts)
        x_t = torch.sqrt(alpha_t_bar)[:, None, None, None] * x_0 + torch.sqrt(1 - alpha_t_bar)[:, None, None, None] * eposilon
        e_hat = self.forward(x_t, ts[:, None].type(torch.float))
        loss = F.mse_loss(e_hat.view(e_hat.shape[0], -1), eposilon.view(eposilon.shape[0], -1))

        return loss

if __name__ == "__main__":
    data = DataSet()
    DataLoader = DataLoader(data, batch_size=4, shuffle=True)
    model = DiffusionModel()

    for i, x in enumerate(DataLoader):
        t = torch.randint(0, 1000, size=(x.shape[0], 1), dtype=torch.int64)
        x = model(x, t)
        
        print(x.shape)
        break

