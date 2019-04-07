import argparse
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
import numpy as np
import time
#import lmdb
from scipy import misc
import cal as c
from utils import mkdir_p

if not os.path.isdir("./results/"):
        mkdir_p("./results/")


parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

parser.add_argument('--in-dataset', default="cifar10", type=str,
                    help='neural network name and training set')
parser.add_argument('--out-dataset', default="Imagenet", type=str,
                    help='out-of-distribution dataset')
parser.add_argument('--magnitude', default=0.0014, type=float,
                    help='perturbation magnitude')
parser.add_argument('--temperature', default=1000, type=int,
                    help='temperature scaling')
parser.add_argument('--gpu', default = '0', type = str,
		    help='gpu index')
parser.add_argument('--wide', dest='wide', action='store_true',
                    help='use wide-resnet')


def main():
    print ("Start")
    global args
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    c.test(args.in_dataset, args.out_dataset, args.wide, args.magnitude, args.temperature)

if __name__ == '__main__':
    main()

















