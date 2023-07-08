'''
Author: Pengbo
Date: 2022-02-23 15:42:01
LastEditTime: 2023-03-01 08:31:15
Description: 

'''
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset

__all__ = ['VisibleInfraredPairH5Dataset']

class VisibleInfraredPairH5Dataset(Dataset):
    def __init__(self, h5file_path, transform, prefix='data/'):
        self.h5file_path = h5file_path
        self.transform = transform
        self.data_list = self._read_all_data_()

    def _read_all_data_(self):
        h5f = h5py.File(self.h5file_path, 'r')
        self.keys = list(h5f['ir_patchs'].keys())
        data_list = []
        for idx in range(self.__len__()):
            key = self.keys[idx]
            ir = np.array(h5f['ir_patchs'][key])
            vis = np.array(h5f['vis_patchs'][key])
            data_list.append((vis, ir))
        h5f.close()
        return data_list

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        vis, ir = self.data_list[index]
        return torch.FloatTensor(vis), torch.FloatTensor(ir)
