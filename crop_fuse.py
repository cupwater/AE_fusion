'''
Author: Peng Bo
Date: 2023-02-04 10:17:01
LastEditTime: 2023-02-04 10:37:04
Description: 

'''


import cv2
import numpy as np
import pdb

ir_click_list = []
vis_click_list = []

dst_pts = np.float32(
        [[0, 0],
        [640, 0],
        [0, 512],
        [640, 512]]
    )

src_pts = np.float32(
   [[368, 137],
   [1583, 137],
   [368, 1075],
   [1583, 1075]]
)



def get_affine_M(src_pts=np.float32([[10, 10], [200, 20],
                                        [30, 250], [250, 250]]) , 
                          dst_pts=np.float32([[0, 0], [512, 0],
                                        [0, 640], [512, 640]])):
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M_reverse = cv2.getPerspectiveTransform(dst_pts, src_pts)
    return np.round(M, 3), np.round(M_reverse, 3)


def crop_tetragon(img, tgt_size, M, M_reverse):
    # img_width, img_height = 512, 640
    dst_img = cv2.warpPerspective(
        img, M, tgt_size,
        #borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    reverse_height, reverse_width = img.shape[:2]
    reverse_img = cv2.warpPerspective(
        dst_img,
        M_reverse, (reverse_width, reverse_height),
        #borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    
    return dst_img, reverse_img


def crop_tetragon_noaffine(img, tgt_size, top_left=(388, 127), bottom_right=(1603, 1065)):
    # img_width, img_height = 512, 640
    crop_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    resize_crop_img = cv2.resize(crop_img, tgt_size)
    return resize_crop_img, crop_img


def on_IR_EVENT_BUTTON(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        # cv2.circle(img, (x,y), 1, (0,0,255), thinkness=-1)
        # cv2.putText(img, xy, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,0), thickness=1)
        # cv2.imshow("ir", img)
        print(x, y)
        ir_click_list.append([x, y])


def on_VIS_EVENT_BUTTON(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        # cv2.circle(img, (x,y), 1, (0,0,255), thinkness=-1)
        # cv2.putText(img, xy, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,0), thickness=1)
        # cv2.imshow("vis", img)
        vis_click_list.append([x, y])
        print(x, y)

def fusion_imgs(img1, img2):
    _, mask = cv2.threshold(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(img1, img1, mask=mask_inv)
    img_fg = cv2.bitwise_and(img2, img2, mask=mask)
    fuse_img = cv2.add(img_bg, img_fg)
    return fuse_img


if __name__ == "__main__":
    
    # cv2.namedWindow("vis")
    # cv2.namedWindow("ir")
    # cv2.setMouseCallback("vis", on_VIS_EVENT_BUTTON)
    # cv2.setMouseCallback("ir", on_IR_EVENT_BUTTON)

    # M = np.float32([
    #         [ 4.92417784e-01, -1.25833390e-01, -1.57544109e+02],
    #         [-3.73122860e-03, 4.65769967e-01,  -4.10213890e+01],
    #         [ 7.10971686e-06, -2.37132253e-04, 1.00000000e+00]
    #     ])
    # M_reverse =np.float32([
    #         [1.99246198e+00,  7.12991077e-01, 3.43148532e+02],
    #         [ 1.50276106e-02, 2.15628043e+00, 9.08211297e+01],
    #         [-1.06023094e-05, 5.06254471e-04, 1.00000000e+00]
    #     ])

    # M = np.float32(
    #    [[0.527,    0.   , -193.844],
    #    [   0.   ,    0.546,  -74.78 ],
    #    [   0.   ,   -0.   ,    1.   ]]
    # )
    # M_reverse = np.float32(
    #    [[ 1.8984375 ,0., 368.],
    #    [ 0. , 1.83203125, 137.],
    #    [ -0. , -0., 1.]]
    # )

    for i in range(6):
        ir = cv2.imread(f"fusion_data/fusion_pairs/hongwai_{i}.jpg")
        vis = cv2.imread(f"fusion_data/fusion_pairs/kejianguang_{i}.jpg")
        tgt_h, tgt_w = ir.shape[:2]

        # ir_click_list = []
        # vis_click_list = []
        # cv2.imshow("ir", ir)
        # cv2.imshow("vis", vis)
        # cv2.waitKey(-1)
        # src_pts = np.float32(vis_click_list)
        # dst_pts = np.float32(ir_click_list)
        # M, M_reverse = get_affine_M(src_pts=src_pts, dst_pts=dst_pts)
        # dst_img, reverse_img = crop_tetragon(vis, (tgt_w, tgt_h), M, M_reverse)
        # fuse_img = fusion_imgs(vis, reverse_img)
        # cv2.imshow("fuse", fuse_img)
        # cv2.waitKey(-1)
        # print(M)
        # print(M_reverse)
        # pdb.set_trace()
        # cv2.imwrite('fusion_data/fuse_img.jpg', fuse_img)

        dst_img, reverse_img = crop_tetragon_noaffine(vis, (tgt_w, tgt_h))
        cv2.imshow('crop_img', dst_img)
        cv2.imshow('ir', ir)
        cv2.waitKey(-1)

    # img = cv2.imread('fusion_data/test_vis.jpg')
    # _, reverse_img = crop_tetragon(img)
    # fuse_img = fusion_imgs(img, reverse_img)
    # cv2.imwrite('fusion_data/fuse_img.jpg', fuse_img)