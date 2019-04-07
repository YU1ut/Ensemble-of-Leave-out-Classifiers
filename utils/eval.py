from __future__ import print_function, absolute_import
from sklearn.metrics import accuracy_score

__all__ = ['accuracy']

def accuracy(gt_list, predict_list, num):
    pred_y = [[] for i in range(num)]
    for i in range(len(gt_list)):
        gt = gt_list[i]
        if gt < num:
            pred_y[gt].append(predict_list[i])
    acc_sum = 0
    for i in range(num):
        acc_sum += accuracy_score([i] * len(pred_y[i]), pred_y[i])
    return acc_sum / num

