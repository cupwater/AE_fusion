'''
Author: Pengbo
Date: 2022-02-23 15:42:01
LastEditTime: 2023-03-06 17:12:53
Description: 

'''
import os
import cv2
import torch
import pdb
import numpy as np
from torch.utils.data import Dataset

__all__ = ['DehazeDataset']

class DehazeDataset (Dataset):

    # --------------------------------------------------------------------------------
    def __init__(self, imgs_list, transform, prefix='data/'):

        self.prefix = prefix
        # read img_list
        self.imgs_list = [l.strip() for l in open(imgs_list).readlines()]
        self.transform = transform

    def __getitem__(self, index):
        p1, p2 = self.imgs_list[index].strip().split(',')
        haze_path, dehaze_path = os.path.join(self.prefix, p1.strip()), os.path.join(self.prefix, p2.strip())

        haze_img   = cv2.imread(haze_path)
        dehaze_img = cv2.imread(dehaze_path)
        if self.transform != None:
            result = self.transform(image=haze_img, mask=dehaze_img)
            haze_img, dehaze_img = result['image'], result['mask']
        haze_img = haze_img.transpose((2,0,1))
        dehaze_img  = dehaze_img.transpose((2,0,1))
        haze_img, dehaze_img = torch.FloatTensor(haze_img), torch.FloatTensor(dehaze_img)
        dehaze_img = (dehaze_img-127.5) / 127.5
        return haze_img, dehaze_img

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
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def TestTransform(final_size=256, crop_size=224):
        return A.Compose([
            A.Resize(final_size, final_size),
            A.CenterCrop(width=crop_size, height=crop_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    transform_train = TrainTransform(crop_size=110, final_size=128)
    transform_test  = TestTransform(crop_size=110, final_size=128)
    trainset = DehazeDataset('./data/train_list.txt', transform_train, 
        prefix="./data/train_vis_dehaze_img_images")
    
    haze_img, dehaze_img = trainset.__getitem__(1)
    import pdb
    pdb.set_trace()
