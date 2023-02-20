'''
Author: Peng Bo
Date: 2022-11-13 22:19:55
LastEditTime: 2023-02-09 11:19:40
Description: 

'''
import time
import cv2
import numpy as np
import torch
import onnxruntime as ort
from models.aod_net import AODnet

# pre-process function for dehaze model
def preprocess(ori_image, input_size):
    image = cv2.resize(ori_image, input_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)/255.0
    image = (image - np.array([0.5, 0.5, 0.5]))
    image = np.divide(image, np.array([0.5, 0.5, 0.5]))
    image = np.expand_dims(np.transpose(image, [2, 0, 1]), axis=0)
    image = image.astype(np.float32)
    return image


# inference using pytorch model
def dehaze_pt(image, pt_model, input_size=(1920, 1080)):  
    processed_frame  = preprocess(image, input_size)
    # inference using pytorch model 
    pt_output = pt_model(torch.FloatTensor(torch.from_numpy(processed_frame)))
    # convert the output to image
    pt_output = pt_output.data.cpu().numpy().squeeze().transpose((1,2,0))*255
    dehaze_image = pt_output.astype(np.uint8)[:,:,::-1]
    return dehaze_image


# inference using onnx model
def dehaze_onnx(image, ort_session, input_size=(1920, 1080)):  
    input_name = ort_session.get_inputs()[0].name
    input  = preprocess(image, input_size)
    onnx_output = ort_session.run(None, {input_name: input})[0][0]
    onnx_putput = onnx_output.transpose((1,2,0))*255
    dehaze_image = onnx_putput.astype(np.uint8)[:,:,::-1]
    return dehaze_image


if __name__ == '__main__':

    pt_model = AODnet()
    pt_model.load_state_dict(torch.load("dehaze_data/AODNet_dehaze.pth", map_location='cpu'))
    pt_model.eval()

    onnx_model = ort.InferenceSession("dehaze_data/AODNet_Dehaze.onnx")
    cap = cv2.VideoCapture('dehaze_data/fog.mp4')

    videoWriter = cv2.VideoWriter('dehaze_data/defog.mp4', cv2.VideoWriter_fourcc('m','p','4','v'), 25, (1920, 1080))

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1920, 1080))
        if not ret:
            break
        # dehaze_image_pt   = dehaze_pt(frame, pt_model)
        dehaze_image_onnx = dehaze_onnx(frame, onnx_model)
        videoWriter.write(dehaze_image_onnx)

        # cv2.imshow("haze image", frame)
        # cv2.imshow("dehaze image using pt_model", dehaze_image_pt)
        cv2.imshow("dehaze image using onnx model", dehaze_image_onnx)

        # key = cv2.waitKey(1)
        # if key==27 or key == ord("q"):
        #     exit(0)

    cap.release()
    videoWriter.release()