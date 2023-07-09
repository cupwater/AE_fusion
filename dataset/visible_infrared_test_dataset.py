'''
Author: Pengbo
Date: 2022-02-23 15:42:01
LastEditTime: 2023-03-01 08:31:15
Description: 

'''
import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset

__all__ = ['VisibleInfraredTestDataset']


def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img


class VisibleInfraredTestDataset(Dataset):
    def __init__(self, imglist_path, transform, prefix='data/'):
        self.data_list = self._read_all_data_(imglist_path)

    def _read_all_data_(self, prefix):
        data_list = []
        for img_name in os.listdir(os.path.join(prefix, "ir")):
            vis_img = image_read_cv2(os.path.join(prefix, "vi", img_name), mode='GRAY')
            ir_img = image_read_cv2(os.path.join(prefix, "ir", img_name),mode='GRAY')

            vis_img = cv2.resize(vis_img, (120, 160))
            ir_img = cv2.resize(ir_img, (120, 160))

            vis_process = vis_img[np.newaxis, ...]/255.0
            ir_process = ir_img[np.newaxis, ...]/255.0

            data_list.append((vis_process, vis_img, ir_process, ir_img))

        return data_list
            
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        vis_process, vis_img, ir_process, ir_img = self.data_list[index]
        return torch.FloatTensor(vis_process), torch.FloatTensor(ir_process), vis_img, ir_img
