'''
Author: Pengbo
Date: 2022-02-23 15:42:01
LastEditTime: 2023-03-02 08:50:22
Description: 

'''
import os
import cv2
import torch
import pdb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

__all__ = ['YCrCbInfraredPairDataset']

class YCrCbInfraredPairDataset (Dataset):

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
        YCrCb = cv2.cvtColor(rgb, cv2.COLOR_BGR2YCR_CB)
        ir  = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        if self.transform != None:
            result = self.transform(image=YCrCb, mask=ir)
            YCrCb, ir = result['image'], result['mask']
        YCrCb = YCrCb.transpose((2,0,1))
        ir = (ir - 127.5) / 127.5
        ir  = np.expand_dims(ir, axis=2).transpose((2,0,1))
        YCrCb, ir = torch.FloatTensor(YCrCb), torch.FloatTensor(ir)
        Yc, Crc, Cbc = YCrCb[0:1], YCrCb[1:2], YCrCb[2:3]
        return Yc, ir, Crc, Cbc

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
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            #A.Normalize(mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5)),
            #A.RandomBrightnessContrast(p=0.1),
            #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def TestTransform(final_size=256, crop_size=224):
        return A.Compose([
            A.Resize(final_size, final_size),
            A.CenterCrop(width=crop_size, height=crop_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            #A.Normalize(mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5)),
            #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    transform_train = TrainTransform(crop_size=110, final_size=128)
    transform_test  = TestTransform(crop_size=110, final_size=128)
    #list_path = '/home/fanao/pby/code/AE_fusion/data/train_list.txt'
    #prefix = '/home/fanao/pby/code/AE_fusion/data/train_vis_ir_images'
    list_path = './data/train_list.txt'
    prefix = './data/train_vis_ir_images'

    import os

    vis_out_prefix = "./data/vis/"
    ir_out_prefix  = "./data/ir/"
    for idx, line in enumerate(open(list_path).readlines()):
        vis_path, ir_path = line.strip().split(',')
        vis_path, ir_path = os.path.join(prefix, vis_path), os.path.join(prefix, ir_path)
        vis_img, ir_img = cv2.imread(vis_path), cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        Yc = cv2.cvtColor(vis_img, cv2.COLOR_BGR2YCR_CB)[:,:,0]
        cv2.imwrite(os.path.join(vis_out_prefix, f"{idx+1}.png"), Yc)
        cv2.imwrite(os.path.join(ir_out_prefix,  f"{idx+1}.png"), ir_img)

    pdb.set_trace()
    trainset = YCrCbInfraredPairDataset(list_path, transform_train, 
        prefix=prefix)
    Yc, ir, Crc, Cbc, rgb = trainset.__getitem__(10)
    pdb.set_trace()
