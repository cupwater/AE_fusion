'''
Author: Pengbo
Date: 2022-03-04 23:49:14
LastEditTime: 2023-02-02 11:03:59
Description: 

'''
import albumentations as A

def TrainTransform(crop_size=224, final_size=256):
    if not isinstance(crop_size, list):
        crop_size = [crop_size, crop_size]
    if not isinstance(final_size, list):
        final_size = [final_size, final_size]
    return A.Compose([
        A.RandomCrop(width=crop_size[0], height=crop_size[1]),
        A.Resize(final_size[0], final_size[1]),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=15),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.1),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

def TestTransform(crop_size=None, final_size=None):
    if not crop_size is None and not isinstance(crop_size, list):
        crop_size = [crop_size, crop_size]
    if not isinstance(final_size, list):
        final_size = [final_size, final_size]
    if crop_size is None:
        return A.Compose([
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    else:
        return A.Compose([
            A.CenterCrop(width=crop_size[0], height=crop_size[1]),
            A.Resize(final_size[0], final_size[1]),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])


if __name__ == "__main__":
    import cv2
    img = cv2.imread('data/1.jpg')
    print(img.shape)
    transforms = TrainTransform()
    new_img = transforms(image=img)['image']
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('new_img', new_img)
    key=cv2.waitKey(-1)
