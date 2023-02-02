'''
Author: Pengbo
Date: 2022-04-07 13:10:47
LastEditTime: 2022-04-07 13:16:14
Description: 

'''
from sklearn import metrics

__all__ = ['f1_score']


def f1_score(ground, predict, average='micro'):
    return metrics.f1_score(ground, predict, average=average)