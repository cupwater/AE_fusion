# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
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
DIDF_Encoder = nn.DataParallel(Restormer_Encoder(dim=model_dim)).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder(dim=model_dim)).to(device)
BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=model_dim, num_heads=8)).to(device)
DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1, dim=model_dim)).to(device)

# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(
    DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

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

        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        feature_V_B, feature_V_D, _ = DIDF_Encoder(data_VIS)
        feature_I_B, feature_I_D, _ = DIDF_Encoder(data_IR)
        data_VIS_hat, _ = DIDF_Decoder(data_VIS, feature_V_B, feature_V_D)
        data_IR_hat, _ = DIDF_Decoder(data_IR, feature_I_B, feature_I_D)

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
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

            fusion.update(0)
            optimizer1.step()  
            optimizer2.step()
        else:  #Phase II
            feature_F_B = BaseFuseLayer(feature_I_B+feature_V_B)
            feature_F_D = DetailFuseLayer(feature_I_D+feature_V_D)
            
            data_Fuse, feature_F = DIDF_Decoder(data_VIS, feature_F_B, feature_F_D)  
            fusionloss, _,_  = criteria_fusion(data_VIS, data_IR, data_Fuse)
            
            loss = fusionloss + coeff_decomp * loss_decomp
            loss.backward()

            fusion.update(fusionloss.item())
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

            optimizer1.step()  
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()

        vis_rec.update(mse_loss_V.item())
        ir_rec.update(mse_loss_I.item())
        vis_gradient.update(Gradient_loss.item())
        decomp.update(loss_decomp.item())
        losses.update(loss.item(), data_VIS.size(0))

        # # Determine approximate time left
        # batches_done = epoch * len(loader['train']) + i
        # batches_left = num_epochs * len(loader['train']) - batches_done
        # time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        # prev_time = time.time()

        # sys.stdout.write(
        #     "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
        #     % (
        #         epoch,
        #         num_epochs,
        #         i,
        #         len(loader['train']),
        #         loss.item(),
        #         time_left,
        #     )
        # )

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
        'DIDF_Encoder': DIDF_Encoder.state_dict(),
        'DIDF_Decoder': DIDF_Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(f"models_channel{model_dim}/CDDFuse_"+timestamp+'.pth'))


