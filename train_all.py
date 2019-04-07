import argparse
import os
import subprocess

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

# DenseNet on CIFAR-10
for fold in range(1,6):
    subprocess.run(f"python train.py -c checkpoints/cifar10_fold_{fold}_dense_checkpoint --fold {fold} --gpu {args.gpu}", shell=True)

# DenseNet on CIFAR-100
for fold in range(1,6):
    subprocess.run(f"python train.py -c checkpoints/cifar100_fold_{fold}_dense_checkpoint --fold {fold} --gpu {args.gpu} --in-dataset cifar100", shell=True)

# WideResNet on CIFAR-10
for fold in range(1,6):
    subprocess.run(f"python train.py -c checkpoints/cifar10_fold_{fold}_wide_checkpoint --fold {fold} --gpu {args.gpu} --wide", shell=True)

# WideResNet on CIFAR-100
for fold in range(1,6):
    subprocess.run(f"python train.py -c checkpoints/cifar100_fold_{fold}_wide_checkpoint --fold {fold} --gpu {args.gpu} --wide --in-dataset cifar100", shell=True)