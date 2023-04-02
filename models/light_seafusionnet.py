# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import time


class ConvBnLeakyRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return torch.tanh(self.conv(x))/2+0.5


class ConvLeakyRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride, dilation=dilation, groups=groups)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class Sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x


class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride, dilation=dilation, groups=groups)

    def forward(self, x):
        return self.conv(x)


class DenseBlock(nn.Module):
    def __init__(self, channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2*channels, channels)

    def forward(self, x):
        x = torch.cat((x, self.conv1(x)), dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        return x


class RGBD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RGBD, self).__init__()
        self.dense = DenseBlock(in_channels)
        self.convdown = Conv1(3*in_channels, out_channels)
        self.sobelconv = Sobelxy(in_channels)
        self.convup = Conv1(in_channels, out_channels)

    def forward(self, x):
        x1 = self.dense(x)
        x1 = self.convdown(x1)
        x2 = self.sobelconv(x)
        x2 = self.convup(x2)
        return F.leaky_relu(x1+x2, negative_slope=0.1)


class SeAFusionNet(nn.Module):
    def __init__(self, out_channel=1):
        super(SeAFusionNet, self).__init__()
        vis_ch = [4, 8, 12]
        inf_ch = [4, 8, 12]
        output = 1
        self.vis_conv = ConvLeakyRelu2d(1, vis_ch[0])
        self.vis_rgbd1 = RGBD(vis_ch[0], vis_ch[1])
        self.vis_rgbd2 = RGBD(vis_ch[1], vis_ch[2])

        self.inf_conv = ConvLeakyRelu2d(1, inf_ch[0])
        self.inf_rgbd1 = RGBD(inf_ch[0], inf_ch[1])
        self.inf_rgbd2 = RGBD(inf_ch[1], inf_ch[2])

        self.decode4 = ConvBnLeakyRelu2d(
            vis_ch[2]+inf_ch[2], vis_ch[1]+vis_ch[1])
        self.decode3 = ConvBnLeakyRelu2d(
            vis_ch[1]+inf_ch[1], vis_ch[0]+inf_ch[0])
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[0]+inf_ch[0], vis_ch[0])
        self.decode1 = ConvBnTanh2d(vis_ch[0], out_channel)

    def forward(self, rgb, ir):
        # split rgb to YCrCb format
        R, G, B = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        Y = 0.257*R + 0.564*G + 0.098*B + 16
        Cb = -0.148*R - 0.291*G + 0.439*B + 128
        Cr = 0.439*R - 0.368*G - 0.071*B + 128

        # normalize
        Y = Y/255.0
        ir = ir/255.0

        # encode for vis
        x_vis_p = self.vis_conv(Y)
        x_vis_p1 = self.vis_rgbd1(x_vis_p)
        x_vis_p2 = self.vis_rgbd2(x_vis_p1)
        # encode for ir
        x_inf_p = self.inf_conv(ir)
        x_inf_p1 = self.inf_rgbd1(x_inf_p)
        x_inf_p2 = self.inf_rgbd2(x_inf_p1)

        # decode
        x = self.decode4(torch.cat((x_vis_p2, x_inf_p2), dim=1))
        x = self.decode3(x)
        x = self.decode2(x)
        x = self.decode1(x)

        # convert to RGB
        outY = 255.0*x
        R = 1.164*(outY-16) + 1.596*(Cr-128)
        G = 1.164*(outY-16) - 0.392*(Cb-128) - 0.813*(Cr-128)
        B = 1.164*(outY-16) + 2.017*(Cb-128)
        fused_img = torch.cat((R, G, B), 1)
        return fused_img


if __name__ == '__main__':

    import numpy as np
    import cv2
    rgb = cv2.imread("fusion_data/test_vis.png").astype(np.float32)
    rgb = cv2.resize(rgb, (640, 512))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    ir  = cv2.imread("fusion_data/test_ir.png", cv2.IMREAD_GRAYSCALE)
    ir  = cv2.resize(ir, (640, 512))
    ir  = np.expand_dims(ir, axis=2).astype(np.float32)
    
    rgb = np.expand_dims(np.transpose(rgb, [2, 0, 1]), axis=0)
    ir  = np.expand_dims(np.transpose(ir, [2, 0, 1]), axis=0)
    rgb_tensor = torch.FloatTensor(torch.from_numpy(rgb))
    ir_tensor  = torch.FloatTensor(torch.from_numpy(ir))

    pt_model = SeAFusionNet()
    pt_model.load_state_dict(torch.load("fusion_data/LightSeAFusion.pth", map_location='cpu'))
    pt_model.eval()
    pt_output = pt_model(rgb_tensor, ir_tensor)

    # convert the output to image
    pt_output = pt_output.data.cpu().numpy().squeeze().transpose((1,2,0))
    pt_output[pt_output>255] = 255
    pt_output[pt_output<0] = 0
    cv2.imwrite("fusion_data/fuse_pt.png", pt_output.astype(np.uint8)[:,:,::-1])
    
    # inference using onnx model 
    import onnxruntime as ort
    onnx_model = ort.InferenceSession("fusion_data/LightSeAFusionRaw.onnx")

    for i in range(20):
        start_time = time.time()
        prediction = onnx_model.run(None, {"rgb": rgb, "ir": ir})[0][0]
        print(f"inference time: {time.time()-start_time}")


    # prediction = prediction.transpose((1,2,0))
    # prediction[prediction>255] = 255
    # prediction[prediction<0] = 0
    # cv2.imwrite("fusion_data/fuse_onnx.png", prediction.astype(np.uint8)[:,:,::-1])


    # target_size  = (720, 480)
    # vis_writer  = cv2.VideoWriter("fusion_data/videos/vis.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 25.0, target_size)
    # ir_writer   = cv2.VideoWriter("fusion_data/videos/ir.mp4", cv2.VideoWriter_fourcc(*'XVID'), 25.0, target_size)
    # fuse_writer = cv2.VideoWriter("fusion_data/videos/fuse.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 25.0, target_size)

    # for i in range(1500, 3900):
    #     rgb_ori = cv2.imread(f"fusion_data/videos/take_2/VIS/VIS_{i}.jpg")
    #     rgb = cv2.resize(rgb_ori.astype(np.float32), (640, 512))
    #     rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    #     rgb = np.expand_dims(np.transpose(rgb, [2, 0, 1]), axis=0)

    #     ir_ori = cv2.imread(f"fusion_data/videos/take_2/IR/IR_{i}.jpg", cv2.IMREAD_GRAYSCALE)
    #     ir_ori = cv2.imread(f"fusion_data/videos/take_2/IR/IR_{i}.jpg")
    #     ir = cv2.resize(ir_ori, (640, 512))
    #     ir = np.expand_dims(ir, axis=2).astype(np.float32)
    #     ir = np.expand_dims(np.transpose(ir, [2, 0, 1]), axis=0)

    #     prediction = onnx_model.run(None, {"rgb": rgb, "ir": ir})[0][0]
    #     prediction = prediction.transpose((1,2,0))
    #     prediction[prediction>255] = 255
    #     prediction[prediction<0] = 0
    #     fused_img = cv2.resize(prediction.astype(np.uint8)[:,:,::-1], target_size)

    #     vis_writer.write(rgb_ori)
    #     ir_writer.write(ir_ori)
    #     fuse_writer.write(fused_img)



