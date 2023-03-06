'''
Author: Peng Bo
Date: 2023-02-02 08:25:04
LastEditTime: 2023-03-02 08:53:54
Description: 

'''
# coding: utf8
import torch
from torch import nn
import pdb

__all__ = ["SDNet", "LightSDNet"]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,  
                    act_fun=nn.LeakyReLU, BN=None, padding=1):
        super(ConvBlock, self).__init__()
        if not BN is None:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                BN(out_channels),
                act_fun()
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                act_fun()
            )

    def forward(self, x):
        return self.conv(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels=1, mid_channels=16):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels,  mid_channels)
        self.conv2 = ConvBlock(mid_channels, mid_channels)
        self.conv3 = ConvBlock(2*mid_channels, mid_channels)
        self.conv4 = ConvBlock(3*mid_channels, mid_channels)

    def forward(self, x):  
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat((x1, x2), 1))
        x4 = self.conv4(torch.cat((x1, x2, x3), 1))
        return torch.cat((x1, x2, x3, x4), 1)


class DisassembleBlock(nn.Module):
    def __init__(self, in_channels=1, mid_channels=16, out_channels=1):
        super(DisassembleBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels,  mid_channels)
        self.conv2 = ConvBlock(mid_channels, mid_channels)
        self.dec_layer = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 3, padding=1),
            nn.Tanh(),
            #nn.Sigmoid()
        )

    def forward(self, x):  
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        out = self.dec_layer(x2)
        return out


class SqueezeNet(nn.Module):
    def __init__(self, vis_channels=1, ir_channels=1, out_channels=1, mid_channels=16):
        super(SqueezeNet, self).__init__()
        self.vis_branch = DenseBlock(vis_channels, mid_channels)
        self.ir_branch  = DenseBlock(ir_channels,  mid_channels)
        self.fuse_layer = nn.Sequential(
            nn.Conv2d(8*mid_channels, out_channels, 3, padding=1),
            nn.Tanh(),
            #nn.Sigmoid()
        )

    def forward(self, vis_in, ir_in):  
        vis_feat = self.vis_branch(vis_in)
        ir_feat  = self.ir_branch(ir_in)
        fused_out = self.fuse_layer(torch.cat((vis_feat, ir_feat), 1))
        return fused_out


class DecomposeNet(nn.Module):
    def __init__(self, in_channels=1, mid_channels=16, vis_channels=1, ir_channels=1):
        super(DecomposeNet, self).__init__()
        self.extract_layer = ConvBlock(in_channels, mid_channels)
        self.vis_branch = DisassembleBlock(mid_channels, mid_channels, vis_channels)
        self.ir_branch  = DisassembleBlock(mid_channels, mid_channels, ir_channels)

    def forward(self, x):  
        mid_feat = self.extract_layer(x)
        vis_out = self.vis_branch(mid_feat)
        ir_out  = self.ir_branch(mid_feat)
        return vis_out, ir_out
    

class SDNet(nn.Module):
    def __init__(self, fuse_channels=1, mid_channels=16, vis_channels=1, ir_channels=1):
        super(SDNet, self).__init__()
        self.squeeze   = SqueezeNet(vis_channels, ir_channels, fuse_channels, mid_channels)
        self.decompose = DecomposeNet(fuse_channels, mid_channels, vis_channels, ir_channels)

    def forward(self, vis_in, ir_in):  
        fused_img       = self.squeeze(vis_in, ir_in)
        vis_out, ir_out = self.decompose(fused_img)
        return fused_img, vis_out, ir_out


class LightSDNet(nn.Module):
    def __init__(self, fuse_channels=1, mid_channels=6, vis_channels=1, ir_channels=1):
        super(LightSDNet, self).__init__()
        self.squeeze   = SqueezeNet(vis_channels, ir_channels, fuse_channels, mid_channels)
        self.decompose = DecomposeNet(fuse_channels, mid_channels, vis_channels, ir_channels)

    def forward(self, vis_in, ir_in):  
        fused_img       = self.squeeze(vis_in, ir_in)
        vis_out, ir_out = self.decompose(fused_img)
        return fused_img, vis_out, ir_out
        #return fused_img
