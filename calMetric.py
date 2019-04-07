import numpy as np
import time
from scipy import misc
import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sns

diff = [0, 1]

def test(indata, outdata, plot=False):
    global diff
    cifar = np.zeros(9000)
    other = np.zeros(9000)
    np.random.seed(3)
    p1 = np.random.permutation(10000)
    for fold in range(1, 6):
        data = pickle.load(open(f"./results/{indata}_{outdata}_{fold}.p", "rb" ))
        cnt = 0
        for i in range(len(data[f'in_pro'])):
            if i in p1[:1000]: continue
            in_probs = data[f'in_pro'][i]
            out_probs = data[f'out_pro'][i]
            in_probs_ = in_probs[np.nonzero(in_probs)]
            in_e = - np.sum(np.log(in_probs_) * in_probs_)
            out_probs_ = out_probs[np.nonzero(out_probs)]
            out_e = - np.sum(np.log(out_probs_) * out_probs_)

            cifar[cnt] += (np.max(in_probs) - in_e)
            other[cnt] += (np.max(out_probs) - out_e)

            cnt += 1
    diff = cifar.tolist() + other.tolist()
    diff = sorted(list(set(diff)))
    cifar, other = np.array(cifar), np.array(other)
    print (f"#All: {len(data[f'in_pro'])} #Cifar: {len(cifar)} #Other: {len(other)}")

    fpr = tpr95(cifar, other)
    error = detection(cifar, other)
    auroc_ = auroc(cifar, other)
    auprin = auprIn(cifar, other)
    auprout = auprOut(cifar, other)
    print("{:31}{:>22}".format("In-distribution dataset:", indata))
    print("{:31}{:>22}".format("Out-of-distribution dataset:", outdata))
    print("")
    print("{:>34}".format("Baseline"))
    print("{:20}{:13.2f} ".format("FPR at TPR 95%:",fpr*100))
    print("{:20}{:13.2f}".format("Detection error:",error*100))
    print("{:20}{:13.2f}".format("AUROC:",auroc_*100))
    print("{:20}{:13.2f}".format("AUPR In:",auprin*100))
    print("{:20}{:13.2f}".format("AUPR Out:",auprout*100))
    if plot:
        if outdata == 'Imagenet':
            name = 'TINc'
        elif outdata == 'Imagenet_resize':
            name = 'TINr'
        elif outdata == 'LSUN':
            name = 'LSUNc'
        elif outdata == 'LSUN_resize':
            name = 'LSUNr'
        plt.figure(figsize=(3,3))
        sns.distplot(cifar, kde=False, rug=False,label="CIFAR-100")
        sns.distplot(other, kde=False, rug=False,label=name)
        plt.legend(loc='upper right')
        # plt.xlim([-0.0075, -0.0055])
        plt.savefig(f"./results/{indata}_{outdata}_out.png",dpi=200,bbox_inches='tight')

def tpr95(X1, Y1):
    #calculate the falsepositive error when tpr is 95%
    total = 0.0
    fpr = 0.0
    for delta in diff:
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += error2
            total += 1
    fprBase = fpr/total

    return fprBase

def auroc(X1, Y1):
    #calculate the AUROC
    aurocBase = 0.0
    fprTemp = 1.0
    for delta in diff:
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        aurocBase += (-fpr+fprTemp)*tpr
        fprTemp = fpr
    aurocBase += fpr * tpr

    return aurocBase

def auprIn(X1, Y1):
    #calculate the AUPR
    precisionVec = []
    recallVec = []
    auprBase = 0.0
    recallTemp = 1.0
    for delta in diff:
        tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        precisionVec.append(precision)
        recallVec.append(recall)
        auprBase += (recallTemp-recall)*precision
        recallTemp = recall
    auprBase += recall * precision

    return auprBase

def auprOut(X1, Y1):
    #calculate the AUPR
    auprBase = 0.0
    recallTemp = 1.0
    for delta in diff[::-1]:
        fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        auprBase += (recallTemp-recall)*precision
        recallTemp = recall
    auprBase += recall * precision
        
    return auprBase



def detection(X1, Y1):
    #calculate the minimum detection error
    errorBase = 1.0
    for delta in diff:
        tpr_ = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorBase = np.minimum(errorBase, (tpr_+error2)/2.0)

    return errorBase
