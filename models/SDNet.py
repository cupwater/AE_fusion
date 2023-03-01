'''
Author: Peng Bo
Date: 2023-02-02 08:25:04
LastEditTime: 2023-02-27 17:16:33
Description: 

'''
# coding: utf8
import torch
from torch import nn
import pdb

__all__ = ["SDNet"]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,  act_fun=nn.PReLU, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            act_fun()
        )

    def forward(self, x):
        return self.conv(x)

class FeatBlock(nn.Module):
    def __init__(self, in_channels=3, mid_channels=16):
        super(FeatBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels,  mid_channels)
        self.conv2 = ConvBlock(mid_channels, mid_channels)
        self.conv3 = ConvBlock(mid_channels, mid_channels)
        self.conv4 = ConvBlock(mid_channels, mid_channels)

    def forward(self, x):  
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat((x1, x2), 1))
        x4 = self.conv4(torch.cat((x1, x2, x3), 1))
        return torch.cat((x1, x2, x3, x4), 1)


class SqueezeNet(nn.Module):
    def __init__(self, vis_channels=3, ir_channels=1, out_channels=3, mid_channels=16):
        super(SqueezeNet, self).__init__()
        self.vis_feat_net = FeatBlock(vis_channels, mid_channels)
        self.ir_feat_net  = FeatBlock(ir_channels,  mid_channels)
        self.fuse_layer = nn.Sequential(
            nn.Conv2d(8*mid_channels, out_channels),
            nn.Tanh()
        )

    def forward(self, vis_input, ir_input):  
        vis_feat = self.vis_feat_net(vis_input)
        ir_feat  = self.ir_feat_net(ir_input)
        fused_out = self.fuse_layer(torch.cat((vis_feat, ir_feat), 1))
        return fused_out