
# coding: utf8

from torch import nn
from .conv_block import ConvBlock

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = ConvBlock(1,  16, is_reflect=True, act_fun=nn.PReLU, padding=1)
        self.conv2 = ConvBlock(16, 16, act_fun=nn.PReLU, padding=1)
        self.conv3 = ConvBlock(16, 16, act_fun=nn.Tanh, padding=1)
        self.conv4 = ConvBlock(16, 16, act_fun=nn.Tanh, padding=1)

    def forward(self, data_train):
        feat1 = self.conv1(data_train)
        feat2 = self.conv2(feat1)
        featB = self.conv3(feat2)
        featD = self.conv4(feat2)
        return feat1, feat2, featB, featD