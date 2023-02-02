# coding: utf8

from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                is_reflect=False, act_fun=nn.PReLU, padding=0):
        super(ConvBlock, self).__init__()
        if is_reflect:
            self.conv = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                act_fun()
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                act_fun()
            )

    def forward(self, x):
        return self.conv(x)