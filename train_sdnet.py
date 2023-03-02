'''
Training script for Image Classification 
Copyright (c) Pengbo, 2021
'''
from __future__ import print_function


import os
import shutil
import time
import yaml
import torch
import torch.onnx
import numpy as np
from skimage.io import imsave
import kornia

import pdb

import models
import dataset
from losses import AdaptiveGradientL2Loss, GradientL2Loss
from torch.nn import functional as F
from augmentation.augment import TrainTransform, TestTransform
from utils import low_pass, Logger, AverageMeter, mkdir_p, progress_bar

state = {}
best_loss = 999
use_cuda = True


def main(config_file, is_eval):
    global state, best_loss, use_cuda

    # parse config of model training
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    common_config = config['common']

    state['lr'] = common_config['lr']
    if not os.path.isdir(common_config['save_path']):
        mkdir_p(common_config['save_path'])
    use_cuda = torch.cuda.is_available()

    data_config = config['dataset']
    transform_train = TrainTransform(
        crop_size=data_config['crop_size'], final_size=data_config['final_size'])
    transform_test = TestTransform(
        crop_size=data_config['crop_size'], final_size=data_config['final_size'])

    print('==> Preparing dataset %s' % data_config['type'])

    # create dataset for training and testing
    trainset = dataset.__dict__[data_config['type']](
        data_config['train_list'], transform_train,
        prefix=data_config['prefix'])
    testset = dataset.__dict__[data_config['type']](
        data_config['test_list'], transform=None,
        prefix=data_config['prefix'])
    # testset = dataset.__dict__[data_config['type']](
    #     data_config['test_list'], transform_test,
    #     prefix=data_config['prefix'])

    # create dataloader for training and testing
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=common_config['train_batch'], shuffle=True, num_workers=5)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=common_config['test_batch'], shuffle=False, num_workers=5)

    # Model
    print("==> creating model '{}'".format(common_config['arch']))
    model = models.__dict__[common_config['arch']]()
    # model.load_state_dict(torch.load(common_config['pretrained_weights'])[
    #                       'state_dict'], strict=False)
    if use_cuda:
        model = model.cuda()
    torch.backends.cudnn.benchmark = True

    # optimizer and scheduler
    optimizer = torch.optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=common_config['lr'],
        momentum=0.9,
        weight_decay=common_config['weight_decay'])

    # logger
    logger = Logger(os.path.join(common_config['save_path'], 'log.txt'))
    logger.set_names(['Learning Rate', 'intensitiy', 'reconstruct', 'gradient', 'loss'])

    if is_eval:
        model.load_state_dict(torch.load(os.path.join(
            common_config['save_path'], 'checkpoint.pth.tar'))['state_dict'], strict=True)
        return

    model = torch.nn.DataParallel(model, device_ids=[0,1,2])

    # Train and val
    for epoch in range(common_config['epoch']):
        adjust_learning_rate(optimizer, epoch, common_config)
        print('\nEpoch: [%d | %d] LR: %f' %
              (epoch + 1, common_config['epoch'], state['lr']))
        intensitiy, reconstruct, gradient, loss = \
                            train(trainloader, model, optimizer, \
                                  use_cuda, epoch, common_config['print_interval'])
        # append logger file
        logger.append([state['lr'], intensitiy, reconstruct, gradient, loss])
        best_loss = min(loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
        }, loss < best_loss, save_path=common_config['save_path'])

    logger.close()


def train(trainloader, model, optimizer, use_cuda, epoch, print_interval=100):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    losses_intensity = AverageMeter()
    losses_reconstruct = AverageMeter()
    losses_gradient = AverageMeter()

    end = time.time()

    criterion_AdapGradLoss = AdaptiveGradientL2Loss()
    model.train()
    for batch_idx, (vis_in, ir_in, Cr, Cb) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            vis_in, ir_in = vis_in.cuda(), ir_in.cuda()

        fused_img, vis_out, ir_out = model(vis_in, ir_in)
        loss_intensity = F.mse_loss(fused_img, vis_in, reduction='mean') + \
                    0.5*F.mse_loss(fused_img, ir_in, reduction='mean')
        loss_reconstruct = F.mse_loss(vis_out, vis_in, reduction='mean') + \
                    F.mse_loss(ir_out, ir_in, reduction='mean')

        loss_gradient = criterion_AdapGradLoss(fused_img, vis_in, ir_in)
        all_loss = loss_intensity + 80 * loss_gradient + loss_reconstruct

        losses_intensity.update(loss_intensity, vis_in.size(0)) 
        losses_reconstruct.update(loss_reconstruct, vis_in.size(0)) 
        losses_gradient.update(loss_gradient, vis_in.size(0)) 
        pdb.set_trace()
        losses.update(all_loss.item(), vis_in.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()

        if batch_idx % print_interval == 0:
            print("iter/epoch: %d / %d \t loss_intensity: %.3f \t loss_reconstruct: %.3f, \
                    loss_gradient: %.3f \t losses: %.3f" % (
                batch_idx, epoch, losses_intensity.avg, losses_reconstruct.avg,
                losses_gradient.avg, losses.avg))

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.2f ' % (losses.avg))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return (losses_intensity.avg, losses_reconstruct.avg, losses_gradient.avg, losses.avg)


def test(testloader, model, use_cuda):
    # switch to evaluate mode
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.eval()
    for batch_idx, (vis_in, ir_in, Cr, Cb) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            vis_in, ir_in = vis_in.cuda(), ir_in.cuda()
        fused_img, _, _ = model(vis_in, ir_in)
        YCrCb = torch.cat((fused_img, Cr, Cb), dim=1)
        for idx in range(fused_img.shape[0]):
            img = np.transpose(YCrCb[idx], (1, 2, 0))
            imsave(f"data/fusion/test_results/{batch_idx}_{idx}.jpg", img)
        progress_bar(batch_idx, len(testloader))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return 


def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(
            save_path, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, config):
    global state
    if epoch in config['scheduler']:
        state['lr'] *= config['gamma']
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    # model related, including  Architecture, path, datasets
    parser.add_argument('--config-file', type=str,
                        default='experiments/template/config.yaml')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    main(args.config_file, args.eval)
