'''
Author: Pengbo
Date: 2022-02-23 15:42:01
LastEditTime: 2023-02-02 09:13:40
Description: 

'''
import os
import cv2
import torch
from torch.utils.data import Dataset

__all__ = ['VisibleInfraredPairDataset']

class VisibleInfraredPairDataset (Dataset):

    # --------------------------------------------------------------------------------
    def __init__(self, imgs_list, transform, prefix='data/'):

        self.prefix = prefix
        # read img_list
        self.imgs_list = [l.strip() for l in open(imgs_list).readlines()]
        self.transform = transform

    def __getitem__(self, index):
        p1, p2 = self.img_list[index].strip().split(',')
        rgb_path, ir_path = os.path.join(self.prefix, p1.strip()), os.path.join(self.prefix, p2.strip())

        rgb = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
        ir  = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

        if self.transform != None:
            rgb = self.transform(rgb)
            ir  = self.transform(ir)

        rgb, ir = rgb.transpose((2, 0, 1)), ir.transpose((2, 0, 1))
        rgb, ir = torch.FloatTensor(rgb), torch.FloatTensor(ir)
        return rgb, ir

    def __len__(self):
        return len(self.imgs_list)
