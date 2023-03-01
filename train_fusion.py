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

import pdb

import models
import dataset
import losses
from augmentation.augment import TrainTransform, TestTransform
from utils import Logger, AverageMeter, mkdir_p, progress_bar

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

    # get all the loss functions into criterion_list
    # optimizer and scheduler
    criterion_list = []
    for loss_key, loss_dict in config['loss_config'].items():
        criterion = losses.__dict__[loss_dict['type']]()
        weight = loss_dict['weight']
        criterion_list.append([criterion, weight, loss_key])

    optimizer = torch.optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=common_config['lr'],
        momentum=0.9,
        weight_decay=common_config['weight_decay'])

    # logger
    logger = Logger(os.path.join(common_config['save_path'], 'log.txt'))
    logger.set_names(['Learning Rate', 'bg_diff', 'detail_diff',
                     'vis_rec', 'ir_rec', 'vis_gradient', 'loss'])

    if is_eval:
        model.load_state_dict(torch.load(os.path.join(
            common_config['save_path'], 'checkpoint.pth.tar'))['state_dict'], strict=True)
        test(testloader, model, criterion_list, use_cuda)
        return

    # Train and val
    for epoch in range(common_config['epoch']):
        adjust_learning_rate(optimizer, epoch, common_config)
        print('\nEpoch: [%d | %d] LR: %f' %
              (epoch + 1, common_config['epoch'], state['lr']))
        bg_diff, detail_diff, vis_rec, ir_rec, vis_gradient, loss = \
                            train(trainloader, model, criterion_list, optimizer, \
                                  use_cuda, epoch, common_config['print_interval'])
        # append logger file
        logger.append([state['lr'], bg_diff, detail_diff,
                      vis_rec, ir_rec, vis_gradient, loss])
        best_loss = min(loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
        }, loss < best_loss, save_path=common_config['save_path'])

    test(testloader, model, criterion_list, use_cuda)
    logger.close()


def train(trainloader, model, criterion_list, optimizer, use_cuda, epoch, print_interval=100):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    bg_diff = AverageMeter()
    detail_diff = AverageMeter()
    vis_rec = AverageMeter()
    ir_rec = AverageMeter()
    vis_gradient = AverageMeter()

    end = time.time()

    model.train()
    for batch_idx, (vis_input, ir_input) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            vis_input, ir_input = vis_input.cuda(), ir_input.cuda()
        out_vis, vis_feat_bg, vis_feat_detail, out_ir, \
            ir_feat_bg, ir_feat_detail = model(vis_input, ir_input)

        all_loss = 0
        for loss_fun, weight, loss_key in criterion_list:
            if loss_key == 'bg_dif':
                temp_loss = weight * \
                    torch.tanh(loss_fun(ir_feat_bg, vis_feat_bg))
                bg_diff.update(temp_loss.item(), vis_input.size(0))
            elif loss_key == 'detail_dif':
                temp_loss = weight * \
                    torch.tanh(loss_fun(ir_feat_detail, vis_feat_detail))
                detail_diff.update(temp_loss)
            elif loss_key == 'vis_rec':
                temp_loss = weight * loss_fun(out_vis, vis_input)
                vis_rec.update(temp_loss)
            elif loss_key == 'ir_rec':
                temp_loss = weight * loss_fun(out_ir, ir_input)
                ir_rec.update(temp_loss)
            elif loss_key == 'vis_gradient':
                temp_loss = weight * loss_fun(vis_input, out_vis)
                vis_gradient.update(temp_loss)
            else:
                print("error, no such loss")
            all_loss += temp_loss

        losses.update(all_loss.item(), vis_input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()

        if batch_idx % print_interval == 0:
            print("iter: %d, epoch: %d, bg_dif: %.3f, detail_dif, %.3f, \
                    vis_rec: %.3f, ir_rec: %.3f, vis_gradient: %.3f, losses: %.3f" % (
                batch_idx, epoch, bg_diff.avg, detail_diff.avg,
                vis_rec.avg, ir_rec.avg, vis_gradient.avg, losses.avg))

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.2f ' % (losses.avg))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return (bg_diff.avg, detail_diff.avg, vis_rec.avg,
            ir_rec.avg, vis_gradient.avg, losses.avg)


def test(testloader, model, criterion, use_cuda):
    global best_loss
    # switch to evaluate mode
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    model.eval()
    for batch_idx, (vis_input, ir_input) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            vis_input, ir_input = vis_input.cuda(), ir_input.cuda()

        fuse_out = model(vis_input, ir_input).cpu().detach().numpy()
        for idx in range(fuse_out.shape[0]):
            img = np.transpose(fuse_out[idx], (1, 2, 0))
            imsave(f"data/fusion/test_results/{batch_idx}_{idx}.jpg", img)

        progress_bar(batch_idx, len(testloader))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return (losses.avg)


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
    import pdb
    pdb.set_trace()
    main(args.config_file, args.eval)
