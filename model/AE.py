# coding: utf8

import torch
from torch import nn
from .encoder import Encoder
from .decoder import Decoder

class AutoEncoder(nn.Module):
    def __init__(self, fuse_mode='Sum'):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fuse_mode = fuse_mode
    
    def forward(self, vis_input, ir_input, is_train=True):

        vis_feat1, vis_feat2, vis_feat_bg, vis_feat_detail = self.encoder(vis_input)
        ir_feat1,  ir_feat2,  ir_feat_bg,  ir_feat_detail  = self.encoder(ir_input)

        if is_train:
            out_vis = self.decoder(vis_feat1, vis_feat2, vis_feat_bg, vis_feat_detail)
            out_ir  = self.decoder(ir_feat1,  ir_feat2,  ir_feat_bg,  ir_feat_detail)
            return out_vis, vis_feat_bg, vis_feat_detail, out_ir, ir_feat_bg, ir_feat_detail

        if self.fuse_mode == 'Sum':
            feat_bg     = ir_feat_bg + vis_feat_bg
            feat_detail = ir_feat_detail + vis_feat_detail

            feat1       = ir_feat1 + vis_feat1
            feat2       = ir_feat2+vis_feat2

        elif self.fuse_mode == 'Average':
            feat_bg = (ir_feat_bg + vis_feat_bg)/2
            feat_detail = (ir_feat_detail + vis_feat_detail)/2

            feat1 = (ir_feat1 + vis_feat1)/2
            feat2 = (ir_feat2 + vis_feat2)/2
        else:
            print('Wrong!')

        out = self.decoder(feat1, feat2, feat_bg, feat_detail)
        return out