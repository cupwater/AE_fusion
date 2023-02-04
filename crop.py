'''
Author: Peng Bo
Date: 2023-02-04 10:17:01
LastEditTime: 2023-02-04 10:24:22
Description: 

'''


import cv2
import numpy as np

"""
    
"""
def get_rotate_crop_image(img, points):
    '''
        input: 
            img: cv2:mat
            points: np:dtype->float32,shape:[4,2]
        output:
            dst_img: cv2:mat
    '''
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])

    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    #if dst_img_height * 1.0 / dst_img_width >= 1.5:
    #    dst_img = np.rot90(dst_img)
    return dst_img


if __name__ == "__main__":
    img = cv2.imread('data/1.jpg')
    points = np.array([
        [10, 10],
        [100, 20],
        [30, 100],
        [150, 150]
    ], dtype=np.float32)
    crop_img = get_rotate_crop_image(img, points)
    cv2.imwrite('data/crop_test.jpg', crop_img)
    # import pdb
    # pdb.set_trace()