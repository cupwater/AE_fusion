
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
from augmentation.transform import MirrorTransform, SpatialTransform


__all__ = ['NonalignedVisibleInfraredPairH5Dataset']

class NonalignedVisibleInfraredPairH5Dataset(Dataset):
    def __init__(self, h5file_path, transform, prefix='data/', is_aug=True):
        self.h5file_path = h5file_path
        self.transform = transform

        # data augmentation
        self.mirror_aug = MirrorTransform()
        self.spatial_aug = SpatialTransform(do_rotation=True,
                                            angle_x=(-np.pi/9, np.pi/9),
                                            angle_y=(-np.pi/9, np.pi/9),
                                            angle_z=(-np.pi/9, np.pi/9),
                                            do_scale=True,
                                            scale=(0.75, 1.25))

        self.data_list = self._read_all_data_()
        self.is_aug = is_aug

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
        fix_img, mov_img = self.data_list[index]
        fix_lab, mov_lab = None, None
        # augmentation
        if self.is_aug:
            code_mir = self.mirror_aug.rand_code()
            code_spa = self.spatial_aug.rand_coords(mov_img.shape[2:])
            mov_img = self.mirror_aug.augment_mirroring(mov_img, code_mir)
            mov_img = self.spatial_aug.augment_spatial(mov_img, code_spa)
            fix_img = self.mirror_aug.augment_mirroring(fix_img, code_mir)
            fix_img = self.spatial_aug.augment_spatial(fix_img, code_spa)

            if mov_lab is not None:
                mov_lab = self.mirror_aug.augment_mirroring(mov_lab, code_mir)
                mov_lab = self.spatial_aug.augment_spatial(mov_lab, code_spa, mode='nearest')
            if fix_lab is not None:
                fix_lab = self.mirror_aug.augment_mirroring(fix_lab, code_mir)
                fix_lab = self.spatial_aug.augment_spatial(fix_lab, code_spa, mode='nearest')

        return torch.FloatTensor(fix_img), torch.FloatTensor(mov_img), \
                torch.FloatTensor(fix_lab), torch.FloatTensor(mov_lab)
