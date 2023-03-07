'''
Author: Peng Bo
Date: 2023-03-06 15:13:05
LastEditTime: 2023-03-06 15:25:24
Description: 

'''
from __future__ import division

# torch condiguration
import argparse
import os
import pdb
from random import uniform
import numpy as np
from PIL import Image
import cv2
import h5py


def read_nyu_depth(nyu_deepth_path):
    nyu_depth = h5py.File(nyu_deepth_path, 'r')
    images = nyu_depth['images']
    depths = nyu_depth['depths']
    return images, depths


def generate_haze(images, depths, prefix, img_size=(240, 320), img_num=1445):
    total_num = 0
    for index in range(img_num):
        gt_image = (images[index, :, :, :]).astype(float)
        gt_image = np.swapaxes(gt_image, 0, 2)
        gt_image = np.array(Image.fromarray(gt_image.astype(np.uint8)).resize(img_size)).astype(np.float32)
        gt_image = gt_image / 255

        gt_depth = depths[index, :, :]
        gt_depth = np.array(Image.fromarray(gt_depth).resize(img_size)).astype(np.float32)
        gt_depth = (gt_depth) / gt_depth.max()
        gt_depth = np.swapaxes(gt_depth, 0, 1)

        for j in range(7):
            for k in range(3):
                bias = 0.05
                temp_beta = 0.4 + 0.2*j
                beta = uniform(temp_beta-bias, temp_beta+bias)
                tx1 = np.exp(-beta * gt_depth)
                #A
                abias = 0.1
                temp_a = 0.5 + 0.2*k
                a = uniform(temp_a-abias, temp_a+abias)
                A = [a,a,a]

                m = gt_image.shape[0]
                n = gt_image.shape[1]

                rep_atmosphere = np.tile(np.reshape(A, [1, 1, 3]), [m, n, 1])
                tx1 = np.reshape(tx1, [m, n, 1])
                max_transmission = np.tile(tx1, [1, 1, 3])
                haze_image = gt_image * max_transmission + rep_atmosphere * (1 - max_transmission)


                total_num = total_num + 1
                out_path = os.path.join(prefix, f"nyu_dehaze/{total_num}.jpg")
                cv2.imwrite(out_path, np.array(gt_image*255.0).astype(np.uint8))
                out_path = os.path.join(prefix, f"nyu_haze/{total_num}.jpg")
                cv2.imwrite(out_path, np.array(haze_image*255.0).astype(np.uint8))


def split_train_val(img_list, prefix, ratio=0.3):
    train_list = np.random.choice(img_list, int(len(img_list)*(1-ratio)), replace=False)
    train_list = [ line + ',' + line.replace('nyu_dehaze', 'nyu_haze') for line in train_list]
    val_list   = list(set(img_list) - set(train_list))
    val_list = [ line.replace('nyu_dehaze', 'nyu_haze') + ',' + line for line in val_list]
    with open(os.path.join(prefix, 'train_list.txt'), 'w') as fout:
        fout.write("\n".join(train_list))
    with open(os.path.join(prefix, 'val_list.txt'), 'w') as fout:
        fout.write("\n".join(val_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nyu-path', type=str, default='./data/dehaze/nyu_depth_v2_labeled.mat')
    parser.add_argument('--prefix', type=str, required=True, help='path to synthesized hazy images dataset store')
    args = parser.parse_args()
    images, depths = read_nyu_depth(args.nyu_path)
    generate_haze(images, depths, args.prefix)
    import glob
    img_list = glob.glob(args.prefix+'/nyu_dehaze/*.jpg')
    split_train_val(img_list, args.prefix)
