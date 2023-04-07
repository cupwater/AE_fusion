'''
Author: Peng Bo
Date: 2023-02-04 10:17:01
LastEditTime: 2023-02-04 10:37:04
Description: 

'''

import cv2

def crop_tetragon_noaffine(img, tgt_size, top_left=(388, 127), bottom_right=(1603, 1065)):
    # img_width, img_height = 512, 640
    crop_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    resize_crop_img = cv2.resize(crop_img, tgt_size)
    return resize_crop_img, crop_img

if __name__ == "__main__":
    

    for i in range(6):
        ir = cv2.imread(f"fusion_data/fusion_pairs/hongwai_{i}.jpg")
        vis = cv2.imread(f"fusion_data/fusion_pairs/kejianguang_{i}.jpg")
        tgt_h, tgt_w = ir.shape[:2]

        dst_img, reverse_img = crop_tetragon_noaffine(vis, (tgt_w, tgt_h))
        cv2.imshow('crop_img', dst_img)
        cv2.imshow('ir', ir)
        cv2.waitKey(-1)
