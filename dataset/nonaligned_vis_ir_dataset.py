
import torch
import numpy as np
import os
import cv2
from torch.utils.data import Dataset
from augmentation.transform import MirrorTransform, SpatialTransform


__all__ = ['NonalignedVisibleInfraredPairDataset']

class NonalignedVisibleInfraredPairDataset(Dataset):
    def __init__(self, imgs_list, transform, prefix='data/', is_aug=True):
        # self.transform = transform
        # data augmentation
        self.mirror_aug = MirrorTransform()
        self.spatial_aug = SpatialTransform(do_rotation=True,
                                            angle_x=(-np.pi/9, np.pi/9),
                                            angle_y=(-np.pi/9, np.pi/9),
                                            angle_z=(-np.pi/9, np.pi/9),
                                            do_scale=True,
                                            scale=(0.75, 1.25))

        self.is_aug = is_aug
        self.imgs_list = [l.strip() for l in open(imgs_list).readlines()]

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

        fix_img, mov_img = rgb / 255.0, ir / 255.0
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
