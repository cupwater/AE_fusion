'''
Author: Peng Bo
Date: 2022-11-13 22:19:55
LastEditTime: 2023-02-06 09:05:43
Description: 

'''
import time
import cv2
import numpy as np
import onnxruntime as ort
from crop_fuse import fusion_imgs, crop_tetragon

def fuse_vis_ir(vis_image, ir_image, ort_session, input_size=(512, 640)):    
    input_name_1 = ort_session.get_inputs()[0].name
    input_name_2 = ort_session.get_inputs()[1].name
    def _preprocess(src_image):
        src_image = cv2.resize(src_image, input_size)
        # pre-process the input image 
        if len(src_image.shape) == 3:
            src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
        input_data = np.expand_dims(src_image, axis=2)
        input_data = (input_data/255.0)
        target_data = np.expand_dims(np.transpose(input_data, [2, 0, 1]), axis=0)
        target_data = target_data.astype(np.float32)
        return target_data

    vis_input = _preprocess(vis_image)
    ir_input  = _preprocess(ir_image)
    fuse_image = ort_session.run(None, {input_name_1: vis_input, input_name_2: ir_input})
    return fuse_image

if __name__ == '__main__':
    onnx_path = "experiments/template/ae_fusion.onnx"
    ort_session = ort.InferenceSession(onnx_path)
    vis_image = cv2.imread("data/test_vis.jpg")
    ir_image  = cv2.imread("data/test_ir.jpg")

    for i in range(20):
        fuse_image = fuse_vis_ir(vis_image, ir_image, ort_session)[0][0][0]
        start_time = time.time()
        _, reverse_img = crop_tetragon(vis_image)
        fuse_img = fusion_imgs(vis_image, reverse_img)
        print("inference time:{}".format(time.time() - start_time))

    from skimage.io import imsave
    imsave(f"data/fuse.jpg", fuse_image)
