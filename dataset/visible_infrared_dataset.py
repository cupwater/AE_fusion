'''
Author: Pengbo
Date: 2022-02-23 15:42:01
LastEditTime: 2023-03-01 08:31:15
Description: 

'''
import os
import cv2
import torch
import pdb
import numpy as np
from PIL import Image
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

        rgb = cv2.imread(rgb_path)
        ir  = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        if self.transform != None:
            result = self.transform(image=rgb, mask=ir)
            rgb, ir = result['image'], result['mask']
        rgb = rgb.transpose((2,0,1))
        ir  = np.tile(np.expand_dims(ir, axis=2), (1,1,3))
        ir  = ir.transpose((2,0,1))
        rgb, ir = torch.FloatTensor(rgb), torch.FloatTensor(ir)
        rgb, ir = rgb / 255.0, ir / 255.0
        return rgb, ir

    def __len__(self):
        return len(self.imgs_list)


if __name__ == "__main__":
    import albumentations as A

    def TrainTransform(final_size=256, crop_size=224):
        return A.Compose([
            A.Resize(final_size, final_size),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=15),
            A.RandomCrop(width=crop_size, height=crop_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def TestTransform(final_size=256, crop_size=224):
        return A.Compose([
            A.Resize(final_size, final_size),
            A.CenterCrop(width=crop_size, height=crop_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    transform_train = TrainTransform(crop_size=110, final_size=128)
    transform_test  = TestTransform(crop_size=110, final_size=128)
    trainset = VisibleInfraredPairDataset('./data/train_list.txt', transform_train, 
        prefix="./data/train_vis_ir_images")
    
    rgb, ir = trainset.__getitem__(1)
    import pdb
    pdb.set_trace()