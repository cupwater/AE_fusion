'''
Author: Peng Bo
Date: 2023-02-04 10:17:01
LastEditTime: 2023-02-04 10:37:04
Description: 

'''


import cv2
import numpy as np


def crop_tetragon(img, src_pts=np.float32([[10, 10], [200, 20],
                                        [30, 250], [250, 250]]) , 
                          dst_pts=np.float32([[0, 0], [512, 0],
                                        [0, 640], [512, 640]])):
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    img_width, img_height = 512, 640
    dst_img = cv2.warpPerspective(
        img,
        M, (img_width, img_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    return dst_img


if __name__ == "__main__":
    img = cv2.imread('data/1.jpg')
    crop_img = crop_tetragon(img)
    cv2.imwrite('data/crop_test.jpg', crop_img)
