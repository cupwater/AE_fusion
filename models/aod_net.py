'''
Author: Peng Bo
Date: 2023-02-15 15:44:37
LastEditTime: 2023-03-07 08:44:41
Description: 

'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class AODnet(nn.Module):   
    def __init__(self):
        super(AODnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,  out_channels=3, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=3,  out_channels=3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6,  out_channels=3, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels=6,  out_channels=3, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, padding=1)
        self.b = 1

    def forward(self, x):  
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3),1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4),1)
        k = F.relu(self.conv5(cat3))
        output = k * x - k + self.b
        return output


class LightAODnet(nn.Module):   
    def __init__(self):
        super(LightAODnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,  out_channels=3, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=3,  out_channels=3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6,  out_channels=3, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=6,  out_channels=3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, padding=1)
        self.b = 1

    def forward(self, x):  
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3),1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4),1)
        k = F.relu(self.conv5(cat3))
        output = k * x - k + self.b
        return output


# input_size = (width, height)
def preprocess(ori_image, input_size=(1920, 1080)):
    image = cv2.resize(ori_image, input_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)/255.0
    image = (image - np.array([0.5, 0.5, 0.5]))
    image = np.divide(image, np.array([0.5, 0.5, 0.5]))
    image = np.expand_dims(np.transpose(image, [2, 0, 1]), axis=0)
    image = image.astype(np.float32)
    return image

if __name__ == '__main__':

    import numpy as np
    import cv2

    ori_img = cv2.imread("dehaze_data/dehaze_input.jpg")
    processed_frame = preprocess(ori_img)

    pt_model = LightAODnet()
    #pt_model.load_state_dict(torch.load("dehaze_data/AODNet_dehaze.pth", map_location='cpu'))
    pt_model.load_state_dict(torch.load("dehaze_data/light_aodnet.pth", map_location='cpu'))
    pt_model.eval()
    # inference using pytorch model 
    pt_output = pt_model(torch.FloatTensor(torch.from_numpy(processed_frame)))

    # convert the output to image
    pt_output = pt_output.data.cpu().numpy().squeeze().transpose((1,2,0))*127.5+127.5
    pt_output[pt_output>255] = 255
    pt_output[pt_output<0] = 0
    cv2.imwrite("dehaze_data/dehaze_output_pt.jpg", pt_output.astype(np.uint8)[:,:,::-1])
    
    # inference using onnx model 
    import onnxruntime as ort
    onnx_model = ort.InferenceSession("dehaze_data/LightAODNet_Dehaze.onnx")
    prediction = onnx_model.run(None, {"input": processed_frame})[0][0]
    prediction = prediction.transpose((1,2,0))*127.5+127.5
    prediction[prediction>255] = 255
    prediction[prediction<0] = 0
    cv2.imwrite("dehaze_data/dehaze_output_onnx.jpg", prediction.astype(np.uint8)[:,:,::-1])

    pdb.set_trace()

