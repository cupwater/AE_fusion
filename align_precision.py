import tensorflow as tf
import torch.nn.functional as F


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


images_vi = tf.random.uniform(shape=[1, 120, 120, 3])
images_ir = tf.random.uniform(shape=[1, 120, 120, 3])
fusion_image = tf.random.uniform(shape=[1, 120, 120, 3])
sept_ir = tf.random.uniform(shape=[1, 120, 120, 3])
sept_vi = tf.random.uniform(shape=[1, 120, 120, 3])


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


loss_intensity = F.mse_loss(fused_img, vis_in, reduction='mean') + \
            0.5*F.mse_loss(fused_img, ir_in, reduction='mean')
loss_reconstruct = F.mse_loss(vis_out, vis_in, reduction='mean') + \
            F.mse_loss(ir_out, ir_in, reduction='mean')

loss_gradient = criterion_AdapGradLoss(fused_img, vis_in, ir_in)
all_loss = loss_intensity + 80 * loss_gradient + loss_reconstruct