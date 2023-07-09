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
        trainset, batch_size=common_config['train_batch'], shuffle=True, num_workers=5)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=common_config['test_batch'], shuffle=False, num_workers=5)

    # Model
    print("==> creating model '{}'".format(common_config['arch']))
    model = models.__dict__[common_config['arch']](dim=16)
    if use_cuda:
        model = model.cuda()
    torch.backends.cudnn.benchmark = True

    # get all the loss functions into criterion_list
    # optimizer and scheduler
    criterion_list = []
    for loss_key, loss_dict in config['loss_config'].items():
        criterion = losses.__dict__[loss_dict['type']]()
        if use_cuda:
            criterion = criterion.cuda()
        weight = loss_dict['weight']
        criterion_list.append([criterion, weight, loss_key])
    optimizer = torch.optim.Adam(model.parameters(), lr=common_config['lr'], \
                                  weight_decay=common_config['weight_decay'])
    # optimizer = torch.optim.SGD(
    #     filter(
    #         lambda p: p.requires_grad,
    #         model.parameters()),
    #     lr=common_config['lr'],
    #     momentum=0.9,
    #     weight_decay=common_config['weight_decay'])

    if is_eval:
        model.load_state_dict(torch.load(os.path.join(
            common_config['save_path'], 'checkpoint.pth.tar'))['state_dict'], strict=True)
        print(test(testloader, model, common_config['save_path'], use_cuda))
        return

    # logger
    logger = Logger(os.path.join(common_config['save_path'], 'log.txt'))
    logger.set_names(['Learning Rate', 'decomp', 'fusion',
                     'vis_rec', 'ir_rec', 'vis_gradient', 'loss',
                    'EN','SD','SF','MI','SCD','VIFF','Qabf','SSIM'])

    # Train and val
    for epoch in range(common_config['epoch']):
        adjust_learning_rate(optimizer, epoch, common_config)
        print('\nEpoch: [%d | %d] LR: %f' %
              (epoch + 1, common_config['epoch'], state['lr']))
        decomp, fusion, vis_rec, ir_rec, vis_gradient, loss = \
                            train(trainloader, model, criterion_list, optimizer, \
                                  use_cuda, epoch, common_config['print_interval'])

        result_metric = test(testloader, model, common_config['save_path'], use_cuda)
        # append logger file
        logger.append([state['lr'], decomp, fusion,
                      vis_rec, ir_rec, vis_gradient, loss] + result_metric.tolist())
        best_loss = min(loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
        }, loss < best_loss, save_path=common_config['save_path'])

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
    decomp = AverageMeter()
    fusion = AverageMeter()

    end = time.time()

    model.train()
    for batch_idx, (vis_input, ir_input) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            vis_input, ir_input = vis_input.cuda(), ir_input.cuda()

        out_vis, vis_feat_bg, vis_feat_detail, out_ir, \
            ir_feat_bg, ir_feat_detail, out_fuse = model(vis_input, ir_input)

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
            elif loss_key == 'decomp':
                temp_loss = weight * loss_fun(vis_feat_bg, vis_feat_detail, ir_feat_bg, ir_feat_detail)
                decomp.update(temp_loss)
            elif loss_key == 'fusion':
                temp_loss = weight * loss_fun(out_vis, out_ir, out_fuse)
                fusion.update(temp_loss)
            else:
                print("error, no such loss")
            all_loss += temp_loss

        losses.update(all_loss.item(), vis_input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        all_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01, norm_type=2)
        optimizer.step()

        if batch_idx % print_interval == 0:
            print("iter/epoch: %d / %d \t decomp: %.3f \t fusion: %.3f, \
                    vis_rec: %.3f \t ir_rec: %.3f \t vis_gradient: %.3f \t losses: %.3f" % (
                batch_idx, epoch, decomp.avg, fusion.avg,
                vis_rec.avg, ir_rec.avg, vis_gradient.avg, losses.avg))

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.2f ' % (losses.avg))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return (decomp.avg, fusion.avg, vis_rec.avg,
            ir_rec.avg, vis_gradient.avg, losses.avg)


def test(testloader, model, save_path, use_cuda):
    global best_loss
    # switch to evaluate mode
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    metric_result = np.zeros((8))

    model.eval()
    for batch_idx, (vis_input, ir_input, vis_imgs, ir_imgs) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            vis_input, ir_input = vis_input.cuda(), ir_input.cuda()
        fuse_out = model(vis_input, ir_input).cpu().detach().numpy()
        fuse_out = 255.0*(fuse_out - np.min(fuse_out)) / (np.max(fuse_out) - np.min(fuse_out))
        fuse_out = np.round(fuse_out)
        vis_imgs = np.round(vis_imgs.numpy())
        ir_imgs = np.round(ir_imgs.numpy())
        for idx in range(fuse_out.shape[0]):
            fuse_img = fuse_out[idx, 0]
            #imsave(f"{save_path}/{batch_idx}_{idx}.jpg", fuse_img)
            metric_result += np.round(np.array([
                            Evaluator.EN(fuse_img), 
                            Evaluator.SD(fuse_img), 
                            Evaluator.SF(fuse_img), 
                            Evaluator.MI(fuse_img, ir_imgs[idx], vis_imgs[idx]),
                            Evaluator.SCD(fuse_img, ir_imgs[idx], vis_imgs[idx]), 
                            Evaluator.VIFF(fuse_img, ir_imgs[idx], vis_imgs[idx]),
                            Evaluator.Qabf(fuse_img, ir_imgs[idx], vis_imgs[idx]), 
                            Evaluator.SSIM(fuse_img, ir_imgs[idx], vis_imgs[idx])]), 3)

        progress_bar(batch_idx, len(testloader))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    metric_result /= len(testloader)
    #print("EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
    #print(str(np.round(metric_result[0], 2))+'\t'
    #        +str(np.round(metric_result[1], 2))+'\t'
    #        +str(np.round(metric_result[2], 2))+'\t'
    #        +str(np.round(metric_result[3], 2))+'\t'
    #        +str(np.round(metric_result[4], 2))+'\t'
    #        +str(np.round(metric_result[5], 2))+'\t'
    #        +str(np.round(metric_result[6], 2))+'\t'
    #        +str(np.round(metric_result[7], 2))
    #        )
    #print("="*80)
    #return (losses.avg)
    return np.round(metric_result, 3)

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
