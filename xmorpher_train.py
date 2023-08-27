'''
Training script for Image Classification 
Copyright (c) Pengbo, 2021
'''
from __future__ import print_function


import os
import shutil
import time
import yaml
import pdb
import torch
import torch.onnx
import numpy as np
from skimage.io import imsave

import models
import dataset
import losses
from augmentation.augment import TrainTransform, TestTransform
from utils import Logger, AverageMeter, mkdir_p, progress_bar, Evaluator

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
    
    print('==> Preparing dataset %s' % data_config['train_type'])
    # create dataset for training and testing
    testset = dataset.__dict__[data_config['test_type']](
        data_config['test_list'], transform_test,
        prefix=data_config['prefix'])
    trainset = dataset.__dict__[data_config['train_type']](
        data_config['train_list'], transform_train,
        prefix=data_config['prefix'])

    # create dataloader for training and testing
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=common_config['train_batch'], shuffle=True, num_workers=16)

    # Model
    print("==> creating model '{}'".format(common_config['arch']))
    model = models.__dict__[common_config['arch']]()
    if use_cuda:
        model = model.cuda()
    model = torch.nn.DataParallel(model)

    torch.backends.cudnn.benchmark = True
    # get all the loss functions into criterion_list
    # optimizer and scheduler

    criterion_dict = {}
    for loss_key, loss_dict in config['loss_config'].items():
        criterion = losses.__dict__[loss_dict['type']]()
        if use_cuda:
            criterion = criterion.cuda()
        criterion_dict[loss_key] =  [criterion, loss_dict['weight']]

    optimizer = torch.optim.Adam(model.parameters(), lr=common_config['lr'], \
                                 weight_decay=common_config['weight_decay'])

    # logger
    logger = Logger(os.path.join(common_config['save_path'], 'log.txt'))
    logger.set_names(['Learning Rate', 'smooth', 'ncc', 'loss',
                    'EN','SD','SF','MI','SCD','VIFF','Qabf','SSIM'])

    # Train and val
    for epoch in range(common_config['epoch']):
        adjust_learning_rate(optimizer, epoch, common_config)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, common_config['epoch'], state['lr']))
        loss, loss_smooth, loss_ncc = train(trainloader, model, criterion_dict, optimizer, \
                                  use_cuda, epoch, common_config['print_interval'])
        result_metric = np.zeros((8))
        # append logger file
        logger.append([state['lr'], loss_smooth, loss_ncc, loss] + result_metric.tolist())
        best_loss = min(loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
        }, loss < best_loss, save_path=common_config['save_path'])
    logger.close()



def train(trainloader, model, criterion_dict, optimizer, use_cuda, epoch, print_interval=100):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    losses_smooth = AverageMeter()
    losses_ncc = AverageMeter()

    end = time.time()
    model.train()
    for batch_idx, (fix_img, mov_img, fix_lab, mov_lab) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            fix_img, mov_img = fix_img.cuda(), mov_img.cuda()
            if fix_lab is not None and mov_lab is not None:
                fix_lab, mov_lab = fix_lab.cuda(), mov_lab.cuda()

        model.zero_grad()
        w_m_to_f, flow = model(mov_img, fix_img, mov_lab, fix_lab)

        all_loss = 0
        for loss_key, (loss_fun, weight) in criterion_dict.items():

            if loss_key == 'smooth':
                temp_loss = loss_fun(flow)
                losses_smooth.update(temp_loss, fix_img.size(0))
            elif loss_key == 'ncc':
                temp_loss = torch.mean(loss_fun(w_m_to_f, fix_img))
                losses_ncc.update(temp_loss.item(), fix_img.size(0))
            all_loss += weight*temp_loss
        all_loss.backward()

        losses.update(all_loss)
        optimizer.step()

    return losses.avg(), losses_smooth.avg(), losses_ncc.avg()


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
        # for optimizer in optimizers_list:
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
