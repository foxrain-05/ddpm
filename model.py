import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import *

class DiffusionModel(nn.Module):
    def __init__(self, t_range=1000, beta_small=1e-4, beta_large=0.02, img_depth=1):
        super().__init__()

        self.t_range = t_range
        self.beta_small = beta_small
        self.beta_large = beta_large

        self.betas = torch.linspace(beta_small, beta_large, t_range)
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

        self.inc = DoubleConv(img_depth, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512 // 2)
        self.up1 = Up(512, 256 // 2)
        self.up2 = Up(256, 128 // 2)
        self.up3 = Up(128, 64)
        self.outc = OutConv(64, img_depth)
        self.sa1 = SAWrapper(256, 8)
        self.sa2 = SAWrapper(256, 4)
        self.sa3 = SAWrapper(128, 8)
    
    def forward(self, x, t):
        x1 = self.inc(x)
        x2 = self.down1(x1) + self.pos_encoding(t, 128, 16)
        x3 = self.down2(x2) + self.pos_encoding(t, 256, 8)
        x3 = self.sa1(x3)
        x4 = self.down3(x3) + self.pos_encoding(t, 256, 4)
        x4 = self.sa2(x4)
        x = self.up1(x4, x3) + self.pos_encoding(t, 128, 8)
        x = self.sa3(x)
        x = self.up2(x, x2) + self.pos_encoding(t, 64, 16)
        x = self.up3(x, x1) + self.pos_encoding(t, 64, 32)
        output = self.outc(x)

        return output
    
    def loss_fn(self, x_0):
        ts = torch.randint(0, self.t_range, size=(x_0.shape[0],), dtype=torch.int64)
        epsilon = torch.randn_like(x_0, dtype=torch.float32)

        noise_imgs = []
        for i in range(len(ts)):
            alpha_t_bar = self.alphas_bar[ts[i]]
            x_t = torch.sqrt(alpha_t_bar) * x_0[i] + torch.sqrt(1 - alpha_t_bar) * epsilon[i]
            noise_imgs.append(x_t)
        
        noise_imgs = torch.stack(noise_imgs, dim=0)

        e_hat = self.forward(noise_imgs, ts[:, None].type(torch.float))

        loss = F.mse_loss(e_hat.view(e_hat.shape[0], -1), epsilon.view(epsilon.shape[0], -1))

        return loss
    
    def sample(self, x, t):
        if t > 1:
            z = torch.randn_like(x, dtype=torch.float32)
        else:
            z = 0

        e_hat = self.forward(x, t.view(1, 1).repeat(x.shape[0], 1))
        pre_scale = 1 / torch.sqrt(self.alphas[t])
        e_scale = (1 - self.alphas[t]) / torch.sqrt(1 - self.alphas_bar[t])
        post_sigma = torch.sqrt(self.betas[t]) * z
        x = pre_scale * (x - e_scale * e_hat) + post_sigma

        return x
    

