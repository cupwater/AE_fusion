'''
Author: Peng Bo
Date: 2023-02-02 08:25:04
LastEditTime: 2023-03-01 08:26:55
Description: 

'''
# coding: utf8
import torch
from torch import nn

__all__ = ["AutoEncoder"]

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


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        #self.conv1 = ConvBlock(1, 16, is_reflect=True, act_fun=nn.PReLU, padding=1)
        self.conv1 = ConvBlock(3,  16, act_fun=nn.PReLU, padding=1)
        self.conv2 = ConvBlock(16, 16, act_fun=nn.PReLU, padding=1)
        self.conv3 = ConvBlock(16, 16, act_fun=nn.Tanh, padding=1)
        self.conv4 = ConvBlock(16, 16, act_fun=nn.Tanh, padding=1)

    def forward(self, data_train):
        feat1 = self.conv1(data_train)
        feat2 = self.conv2(feat1)
        featB = self.conv3(feat2)
        featD = self.conv4(feat2)
        return feat1, feat2, featB, featD


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = ConvBlock(32, 16, act_fun=nn.PReLU, padding=1)
        self.conv2 = ConvBlock(32, 16, act_fun=nn.PReLU, padding=1)
        self.conv3 = ConvBlock(32, 3,  act_fun=nn.Sigmoid, padding=1)

    def forward(self, feat1, feat2, featB, featD):
        out = self.conv1(torch.cat([featB, featD], 1))
        out = self.conv2(torch.cat([out, feat2], 1))
        out = self.conv3(torch.cat([out, feat1], 1))
        return out

class AutoEncoder(nn.Module):
    def __init__(self, fuse_mode='Sum'):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fuse_mode = fuse_mode
    
    def forward(self, vis_input, ir_input):

        vis_feat1, vis_feat2, vis_feat_bg, vis_feat_detail = self.encoder(vis_input)
        ir_feat1,  ir_feat2,  ir_feat_bg,  ir_feat_detail  = self.encoder(ir_input)

        if self.training:
            out_vis = self.decoder(vis_feat1, vis_feat2, vis_feat_bg, vis_feat_detail)
            out_ir  = self.decoder(ir_feat1,  ir_feat2,  ir_feat_bg,  ir_feat_detail)
            return out_vis, vis_feat_bg, vis_feat_detail, out_ir, ir_feat_bg, ir_feat_detail
        else:
            if self.fuse_mode == 'Sum':
                feat_bg     = ir_feat_bg + vis_feat_bg
                feat_detail = ir_feat_detail + vis_feat_detail

                feat1       = ir_feat1 + vis_feat1
                feat2       = ir_feat2 + vis_feat2

            elif self.fuse_mode == 'Average':
                feat_bg = (ir_feat_bg + vis_feat_bg)/2
                feat_detail = (ir_feat_detail + vis_feat_detail)/2

                feat1 = (ir_feat1 + vis_feat1)/2
                feat2 = (ir_feat2 + vis_feat2)/2
            else:
                print('Wrong!')

            out = self.decoder(feat1, feat2, feat_bg, feat_detail)
            return out
