# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

from cddfuse import CDDFuseLight
from dataset import H5Dataset
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
import sys
import time
import datetime
from logger import Logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loss import Fusionloss, cc
import kornia
import pdb


'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
criteria_fusion = Fusionloss()
model_str = 'CDDFuse'

# . Set the hyper-parameters for training
num_epochs = 50 # total epoch
epoch_gap = 16  # epoches of Phase I 

lr = 1e-4
weight_decay = 0
batch_size = 16
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
# Coefficients of the loss function
coeff_mse_loss_VF = 1. # alpha1
coeff_mse_loss_IF = 1.
coeff_decomp = 2.      # alpha2 and alpha4
coeff_tv = 5.

clip_grad_norm_value = 0.01
optim_step = 8
optim_gamma = 0.5



model_dim = 16
# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = nn.DataParallel(CDDFuseLight()).to(device)

# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    model.module.encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    model.module.decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    model.module.base_fuse.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(
    model.module.detail_fuse.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)

MSELoss = nn.MSELoss()  
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIMLoss(window_size=11, reduction='mean')

# data loader
trainloader = DataLoader(H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=16)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# logger
logger = Logger(os.path.join(f"models_channel{model_dim}", 'log.txt'))
logger.set_names(['decomp', 'fusion', 'vis_rec', 'ir_rec', 'vis_gradient', 'loss'])

DIDF_Encoder.train()
DIDF_Decoder.train()
BaseFuseLayer.train()
DetailFuseLayer.train()

for epoch in range(num_epochs):

    losses = AverageMeter()
    vis_rec = AverageMeter()
    ir_rec = AverageMeter()
    vis_gradient = AverageMeter()
    decomp = AverageMeter()
    fusion = AverageMeter()

    ''' train '''
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

        model.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        data_VIS_hat, feature_V_B, feature_V_D, \
                    data_IR_hat, feature_I_B, feature_I_D, \
                    data_Fuse = model(data_VIS, data_IR)        


        cc_loss_B = cc(feature_V_B, feature_I_B)
        cc_loss_D = cc(feature_V_D, feature_I_D)
        mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
        mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)

        Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                               kornia.filters.SpatialGradient()(data_VIS_hat))
        loss_decomp =  (cc_loss_D) ** 2/ (1.01 + cc_loss_B)  

        if epoch < epoch_gap: #Phase I
            fusionloss = 0
            loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * \
                   mse_loss_I + coeff_decomp * loss_decomp + coeff_tv * Gradient_loss
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

            fusion.update(0)
            optimizer1.step()  
            optimizer2.step()
        else:  #Phase II
            fusionloss, _,_  = criteria_fusion(data_VIS, data_IR, data_Fuse)
            
            loss = fusionloss + coeff_decomp * loss_decomp
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

            fusion.update(fusionloss.item())

            optimizer1.step()  
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()

        vis_rec.update(mse_loss_V.item())
        ir_rec.update(mse_loss_I.item())
        vis_gradient.update(Gradient_loss.item())
        decomp.update(loss_decomp.item())
        losses.update(loss.item(), data_VIS.size(0))

    # adjust the learning rate
    scheduler1.step()  
    scheduler2.step()
    if not epoch < epoch_gap:
        scheduler3.step()
        scheduler4.step()

    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6:
        optimizer4.param_groups[0]['lr'] = 1e-6
    
    logger.append([decomp.avg, fusion.avg, vis_rec.avg,
            ir_rec.avg, vis_gradient.avg, losses.avg])

logger.close()

if True:
    checkpoint = {
        'model': model.state_dict(),
    }
    torch.save(checkpoint, os.path.join(f"models_channel{model_dim}/MyCDDFuse_"+timestamp+'.pth'))


