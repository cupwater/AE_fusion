'''
Training script for Image Classification 
Copyright (c) Pengbo, 2021
'''
from __future__ import print_function


import pdb
import cv2
import numpy as np
import os
import shutil
import time
import yaml
import torch
import torch.onnx
from skimage.io import imsave

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
        data_config['test_list'], transform_test,
        prefix=data_config['prefix'])

    # create dataloader for training and testing
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=common_config['train_batch'], shuffle=True, num_workers=5)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=common_config['test_batch'], shuffle=False, num_workers=5)

    # Model
    print("==> creating model '{}'".format(common_config['arch']))
    model = models.__dict__[common_config['arch']]()

    if use_cuda:
        model = model.cuda()
    torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=common_config['lr'],
        momentum=0.9,
        weight_decay=common_config['weight_decay'])

    criterion = losses.L2Loss()

    if is_eval:
        #model = torch.nn.DataParallel(model, device_ids=[1,2])
        model.load_state_dict(torch.load(os.path.join(
            common_config['save_path'], 'checkpoint.pth.tar'))['state_dict'], strict=True)
        model.eval()

        model_path = "dehaze_tiny_aodnet.onnx"
        inputs = torch.randn(1, 3, 1080, 1920) #.to("cuda")
        model = model.cpu()
        torch.onnx.export(model, inputs, model_path, input_names=["input"],
                                output_names=["output"],  verbose=False, opset_version=11)
        import onnx_tool
        onnx_tool.model_profile(model_path, None, None) # pass file name
        return


    # logger
    logger = Logger(os.path.join(common_config['save_path'], 'log.txt'))
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss'])

    # Train and val
    for epoch in range(common_config['epoch']):
        adjust_learning_rate(optimizer, epoch, common_config)
        print('\nEpoch: [%d | %d] LR: %f' %
              (epoch + 1, common_config['epoch'], state['lr']))
        train_loss = train(trainloader, model, criterion, optimizer, use_cuda)
        # append logger file
        logger.append([state['lr'], train_loss, train_loss])
        best_loss = min(train_loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
        }, train_loss < best_loss, save_path=common_config['save_path'])

    test(testloader, model, criterion, use_cuda)
    logger.close()


def train(trainloader, model, criterion, optimizer, use_cuda):
    # switch to train mode
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    model.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        #img = inputs.detach().cpu().numpy()[0]
        #tgt = targets.detach().cpu().numpy()[0]
        #cv2.imwrite('inp.jpg', np.transpose((127.5*img+127.5), (1,2,0)).astype(np.uint8))
        #cv2.imwrite('tgt.jpg', np.transpose((127.5*tgt+127.5), (1,2,0)).astype(np.uint8))
        #pdb.set_trace()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.2f ' % (losses.avg))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return (losses.avg)


def test(testloader, model, criterion, use_cuda):
    global best_loss
    # switch to evaluate mode
    model.eval()
    losses = AverageMeter()
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        progress_bar(batch_idx, len(testloader), 'Loss: %.2f ' % (losses.avg))
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
    main(args.config_file, args.eval)
