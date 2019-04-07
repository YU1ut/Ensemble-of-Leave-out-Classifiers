import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from scipy import misc
import calData as d
import calMetric as m
import models


from PIL import Image
import os
import pandas as pd

start = time.time()
#loading data sets

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
])

criterion = nn.CrossEntropyLoss()


def test(in_dataset, out_dataset, wide, epsilon, temperature):

    testsetout = torchvision.datasets.ImageFolder(os.path.expanduser("./data/{}".format(out_dataset)), transform=transform)
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=100, shuffle=False, num_workers=2)

    if in_dataset == "cifar100": 
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    elif in_dataset == "cifar10":
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    for fold in range(1,6):
        print (f"Processing fold {fold}")

        nclasses = int(in_dataset[5:])
        if wide: 
            net = models.WideResNet(int(nclasses*4/5))
            ck = torch.load(f"./checkpoints/{in_dataset}_fold_{fold}_wide_checkpoint/model_best.pth.tar")
        else:
            net = models.DenseNet(int(nclasses*4/5))
            ck = torch.load(f"./checkpoints/{in_dataset}_fold_{fold}_dense_checkpoint/model_best.pth.tar")
            
        net.load_state_dict(ck['state_dict'])

        net.cuda()
        net.eval()

        d.testData(net, criterion, testloaderIn, testloaderOut, in_dataset, out_dataset, epsilon, temperature, fold)

    m.test(in_dataset, out_dataset, plot=True)
