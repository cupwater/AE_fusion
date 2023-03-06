import tensorflow as tf
from torch.nn import functional as F
from torch import nn
import torch
from losses.loss import AdaptiveGradientL2Loss
import numpy as np

def gradient(input):
    filter = tf.reshape(tf.constant(
        [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]), [3, 3, 1, 1])
    d = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
    return d


def low_pass(input):
    filter = tf.reshape(tf.constant([[0.0947, 0.1183, 0.0947], [
                        0.1183, 0.1478, 0.1183], [0.0947, 0.1183, 0.0947]]), [3, 3, 1, 1])
    d = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
    return d


def low_pass_pt(input):
    weight = torch.nn.Parameter(torch.FloatTensor(
                [[0.0947,0.1183,0.0947],
                    [0.1183,0.1478,0.1183],
                    [0.0947,0.1183,0.0947]]).reshape(1,1,3,3))
    return F.conv2d(input, weight, padding=1)

def gradient_pt(input):
    weight = torch.nn.Parameter(torch.FloatTensor(
                [[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]])).reshape(1,1,3,3)
    return F.conv2d(input, weight, padding=1)


class AdaptiveGradientL2Loss(nn.Module):
    def __init__(self):
        super(AdaptiveGradientL2Loss, self).__init__()

    def forward(self, fused_img, vis, ir):
        vis_grad_lowpass = torch.abs(gradient_pt(low_pass_pt(vis)))
        ir_grad_lowpass  = torch.abs(gradient_pt(low_pass_pt(ir)))
        vis_score = torch.sign(
            vis_grad_lowpass - torch.minimum(vis_grad_lowpass, ir_grad_lowpass))
        ir_score = 1 - vis_score

        fused_img_grad = gradient_pt(fused_img)
        vis_grad = gradient_pt(vis)
        ir_grad = gradient_pt(ir)

        loss_gradient = torch.mean(torch.mul(vis_score.detach(), torch.square(fused_img_grad - vis_grad))) \
                        + torch.mean(torch.mul(ir_score.detach(), torch.square(fused_img_grad - ir_grad)))
        return loss_gradient, vis_score, ir_score


images_vi = tf.random.uniform(shape=[1, 120, 120, 1])
images_ir = tf.random.uniform(shape=[1, 120, 120, 1])
fusion_image = tf.random.uniform(shape=[1, 120, 120, 1])
sept_ir = tf.random.uniform(shape=[1, 120, 120, 1])
sept_vi = tf.random.uniform(shape=[1, 120, 120, 1])


Image_vi_grad = tf.abs(gradient(images_vi))
Image_ir_grad = tf.abs(gradient(images_ir))
Image_vi_weight_lowpass = tf.abs(gradient(low_pass(images_vi)))
Image_ir_weight_lowpass = tf.abs(gradient(low_pass(images_ir)))

Image_vi_score_2 = tf.sign(Image_vi_weight_lowpass-tf.minimum(Image_vi_weight_lowpass, Image_ir_weight_lowpass))
Image_ir_score_2 = tf.sign(Image_ir_weight_lowpass-tf.minimum(Image_vi_weight_lowpass, Image_ir_weight_lowpass))
Image_vi_score = Image_vi_score_2
Image_ir_score = 1-Image_vi_score

g_loss_int = tf.reduce_mean(tf.square(fusion_image - images_ir)) + \
                    0.5 * tf.reduce_mean(tf.square(fusion_image - images_vi))
g_loss_grad = tf.reduce_mean(Image_ir_score * tf.square(gradient(fusion_image) - gradient(images_ir))) + \
                    tf.reduce_mean(Image_vi_score * tf.square(gradient(fusion_image) - gradient(images_vi)))
g_loss_sept = tf.reduce_mean(tf.square(sept_ir - images_ir)) + \
                    tf.reduce_mean(tf.square(sept_vi - images_vi))
g_loss_2 = g_loss_int+80*g_loss_grad+1*g_loss_sept


import pdb
criterion_AdapGradLoss = AdaptiveGradientL2Loss()

fusion_image_pt = np.transpose(fusion_image.numpy(), (0,3,1,2))
fusion_image_pt = torch.FloatTensor(fusion_image_pt)

images_vi_pt = np.transpose(images_vi.numpy(), (0,3,1,2))
images_vi_pt = torch.FloatTensor(images_vi_pt)
sept_vi_pt = np.transpose(sept_vi.numpy(), (0,3,1,2))
sept_vi_pt = torch.FloatTensor(sept_vi_pt)

images_ir_pt = np.transpose(images_ir.numpy(), (0,3,1,2))
images_ir_pt = torch.FloatTensor(images_ir_pt)
sept_ir_pt = np.transpose(sept_ir.numpy(), (0,3,1,2))
sept_ir_pt = torch.FloatTensor(sept_ir_pt)

pdb.set_trace()
loss_intensity = F.mse_loss(fusion_image_pt, images_vi_pt, reduction='mean') + \
            0.5*F.mse_loss(fusion_image_pt, images_ir_pt, reduction='mean')
loss_reconstruct = F.mse_loss(sept_vi_pt, images_vi_pt, reduction='mean') + \
            F.mse_loss(sept_ir_pt, images_ir_pt, reduction='mean')

loss_gradient, vi_score, ir_score = criterion_AdapGradLoss(fusion_image_pt, images_vi_pt, images_ir_pt)
all_loss = loss_intensity + 80 * loss_gradient + loss_reconstruct
pdb.set_trace()
