'''
Author: Pengbo
Date: 2022-02-23 15:42:01
LastEditTime: 2023-02-02 10:30:58
Description: 

'''
import os
import cv2
from PIL import Image
import torch
import pdb
import numpy as np
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
        p1, p2 = self.imgs_list[index].strip().split(',')
        rgb_path, ir_path = os.path.join(self.prefix, p1.strip()), os.path.join(self.prefix, p2.strip())

        rgb = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
        ir  = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

        concat_img = cv2.merge([rgb, rgb, ir]).astype(np.float32)
        if self.transform != None:
            pdb.set_trace()
            rgb = self.transform(rgb)
            ir  = self.transform(ir)

        rgb, ir = rgb.transpose((2, 0, 1)), ir.transpose((2, 0, 1))
        rgb, ir = torch.FloatTensor(rgb), torch.FloatTensor(ir)
        return rgb, ir

    def __len__(self):
        return len(self.imgs_list)


if __name__ == "__main__":
    from torchvision import transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(128),
        transforms.Resize(128),
        transforms.ToTensor(),
    ])
    trainset = VisibleInfraredPairDataset('data/train_list.txt', transform_train, 
        prefix="data/train_vis_ir_images")
    
    trainset.__getitem__(1)