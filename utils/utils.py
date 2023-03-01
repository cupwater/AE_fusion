'''
Author: Peng Bo
Date: 2023-03-01 17:51:23
LastEditTime: 2023-03-01 18:04:44
Description: 

'''
from torch.nn import functional as F
import torch

def low_pass(input):
    weight = torch.nn.Parameter(torch.FloatTensor(
                [[0.0947,0.1183,0.0947],
                    [0.1183,0.1478,0.1183],
                    [0.0947,0.1183,0.0947]]).reshape(1,1,3,3))
    return F.conv2d(input, weight, padding=1)

