'''
Author: Peng Bo
Date: 2023-02-15 15:44:37
LastEditTime: 2023-03-07 08:44:41
Description: 

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
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


class TinyAODnet(nn.Module):   
    def __init__(self):
        super(TinyAODnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,  out_channels=3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=3,  out_channels=3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6,  out_channels=3, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=6,  out_channels=3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=1, padding=0)
        self.b = 1
        self.weights_init_normal()

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


class XLTinyAODnet(nn.Module):   
    def __init__(self):
        super(XLTinyAODnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,  out_channels=3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=3,  out_channels=3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6,  out_channels=3, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=6,  out_channels=3, kernel_size=1, padding=0)
        self.b = 1
        self.weights_init_normal()

    def forward(self, x):  
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3),1)
        k = F.relu(self.conv4(cat2))
        output = k * x - k + self.b
        return output

    ## takes in a module and applies the specified weight initialization
    def weights_init_normal(self):
        '''Takes in a module and initializes all linear layers with weight
           values taken from a normal distribution.'''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)


class XLTinyAODnetPre(nn.Module):   
    def __init__(self):
        super(XLTinyAODnetPre, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,  out_channels=3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=3,  out_channels=3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6,  out_channels=3, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=6,  out_channels=3, kernel_size=1, padding=0)
        self.post_conv = torch.nn.Conv2d(in_channels=3,  out_channels=3, kernel_size=1, groups=3)
        self.b = 1

        self.weights_init_normal()
        self.post_conv.weight = torch.nn.Parameter(127.5*torch.ones(3, 1, 1, 1))
        self.post_conv.bias   = torch.nn.Parameter(127.5*torch.ones(3))

    def forward(self, x):  
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3),1)
        k = F.relu(self.conv4(cat2))
        output = k * x - k + self.b
        output = self.post_conv(output)
        return output

    ## takes in a module and applies the specified weight initialization
    def weights_init_normal(self):
        '''Takes in a module and initializes all linear layers with weight
           values taken from a normal distribution.'''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

# input_size = (width, height)
def preprocess(ori_image, input_size=(1920, 1080)):
    image = cv2.resize(ori_image, input_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = (image - np.array([127.5, 127.5, 127.5]))
    image = np.divide(image, np.array([127.5, 127.5, 127.5]))
    image = np.expand_dims(np.transpose(image, [2, 0, 1]), axis=0)
    image = image.astype(np.float32)
    return image

if __name__ == '__main__':

    import numpy as np
    import cv2

    ori_img = cv2.imread("dehaze_data/dehaze_input.jpg")
    processed_frame = preprocess(ori_img)

    pt_model = XLTinyAODnetPre()
    # pt_model.load_state_dict(torch.load("dehaze_data/dehaze_xltiny_aodnet.pth", map_location='cpu'), strict=False)
    # torch.save(pt_model.state_dict(), "dehaze_data/dehaze_xltiny_aodnet_wopost.pth")
    pt_model.load_state_dict(torch.load("dehaze_data/dehaze_xltiny_aodnet_wopost.pth", map_location='cpu'))
    pt_model.eval()

    # inference using pytorch model 
    pt_output = pt_model(torch.FloatTensor(torch.from_numpy(processed_frame)))
    # convert the output to image
    pt_output = pt_output.data.cpu().numpy().squeeze().transpose((1,2,0))
    pt_output[pt_output>255] = 255
    pt_output[pt_output<0] = 0
    cv2.imwrite("dehaze_data/dehaze_output_pt.jpg", pt_output.astype(np.uint8)[:,:,::-1])
    
    # inference using onnx model 
    import onnxruntime as ort
    onnx_model = ort.InferenceSession("dehaze_data/dehaze_xltiny_aodnet_wopost.onnx")
    for i in range(20):
        start_time = time.time()
        prediction = onnx_model.run(None, {"input": processed_frame})[0][0]
        print(f"inference time: {time.time()-start_time}")
    prediction = prediction.transpose((1,2,0))
    prediction[prediction>255] = 255
    prediction[prediction<0] = 0
    cv2.imwrite("dehaze_data/dehaze_output_onnx.jpg", prediction.astype(np.uint8)[:,:,::-1])