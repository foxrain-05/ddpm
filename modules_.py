import torch
import torch.nn as nn
import torch.nn.functional as F


class InConv(nn.Module):
    def __init__(self, out_channels):
        super(InConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False):
        super(ConvBlock, self).__init__()
        self.residual = residual

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.GELU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.conv_block(x))
        else:
            return self.conv_block(x)
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels, in_channels, residual=True),
            nn.GELU(),
            ConvBlock(in_channels, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.down(x)
    

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(in_channels, in_channels, residual=True),
            nn.GELU(),
            ConvBlock(in_channels, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.up(x)
    

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

if __name__ == '__main__':
    x = torch.randn(1, 1, 64, 64)