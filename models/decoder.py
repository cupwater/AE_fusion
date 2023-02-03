# coding: utf8

import torch
from torch import nn
from .conv_block import ConvBlock

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = ConvBlock(32, 16, act_fun=nn.PReLU, padding=1)
        self.conv2 = ConvBlock(32, 16, act_fun=nn.PReLU, padding=1)
        self.conv3 = ConvBlock(32, 1,  act_fun=nn.Sigmoid, padding=1)

    def forward(self, feat1, feat2, featB, featD):
        out = self.conv1(torch.cat([featB, featD], 1))
        out = self.conv2(torch.cat([out, feat2], 1))
        out = self.conv3(torch.cat([out, feat1], 1))
        return out
