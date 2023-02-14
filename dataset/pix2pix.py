import glob
import os
import numpy as np
from torch.utils.data import Dataset
import h5py


__all__ = ["pix2pix"]

class pix2pix(Dataset):
    def __init__(self, root, transform=None, seed=None):
        imgs = self.make_dataset(root)
        self.root = root
        self.imgs = imgs
        self.transform = transform

        if seed is not None:
            np.random.seed(seed)

    def __getitem__(self, index):
        path = self.imgs[index]
        f = h5py.File(path, 'r')

        haze_image = f['haze'][:]
        GT = f['gt'][:]

        haze_image = np.swapaxes(haze_image, 0, 2)
        GT = np.swapaxes(GT, 0, 2)

        haze_image = np.swapaxes(haze_image, 1, 2)
        GT = np.swapaxes(GT, 1, 2)
        return haze_image, GT

    def __len__(self):
        train_list = glob.glob(self.root+'/*h5')
        return len(train_list)

    def make_dataset(self, dir):
        images = []
        if not os.path.isdir(dir):
            raise Exception('Check dataroot')
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                images.append(os.path.join(dir, fname))
        return images
