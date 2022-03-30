from __future__ import absolute_import
import torch
from PIL import Image
import os
import pandas as pd
import math
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import torchvision
from torchvision import transforms


__all__ = ['CIFAR10Mix', 'CIFAR100Mix']

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def make_dataset(dir):
    images = []
    dir = os.path.expanduser(dir)
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

class CIFAR10Mix(torchvision.datasets.CIFAR10):

    def __init__(self, root, out_path, train=False, val=False,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10Mix, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.outpath = make_dataset(out_path)
        if val:
            #ID Data
            np.random.seed(3)
            p1 = np.random.permutation(len(self.data))
            self.data = self.data[p1[:1000]]
            self.targets = [self.targets[i] for i in p1.tolist()[:1000]]
            #OOD Data
            np.random.seed(3)
            p2 = np.random.permutation(len(self.outpath))
            self.outpath = [self.outpath[i] for i in p2.tolist()[:1000]]
        else:
            #ID Data
            np.random.seed(3)
            p1 = np.random.permutation(len(self.data))
            self.data = self.data[p1[1000:]]
            self.targets = [self.targets[i] for i in p1.tolist()[1000:]]
            #OOD Data
            np.random.seed(3)
            p2 = np.random.permutation(len(self.outpath))
            self.outpath = [self.outpath[i] for i in p2.tolist()[1000:len(p1)]]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if index < len(self.data):
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
        else:
            img_path, target = self.outpath[index - len(self.data)], -1
            img = pil_loader(img_path)
            img = transforms.Resize(32)(img)


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data) + len(self.outpath)


class CIFAR100Mix(torchvision.datasets.CIFAR100):

    def __init__(self, root, out_path, train=False, val=False,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR100Mix, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.outpath = make_dataset(out_path)
        if val:
            #ID Data
            np.random.seed(3)
            p1 = np.random.permutation(len(self.data))
            self.data = self.data[p1[:1000]]
            self.targets = [self.targets[i] for i in p1.tolist()[:1000]]
            #OOD Data
            np.random.seed(3)
            p2 = np.random.permutation(len(self.outpath))
            self.outpath = [self.outpath[i] for i in p2.tolist()[:1000]]
        else:
            #ID Data
            np.random.seed(3)
            p1 = np.random.permutation(len(self.data))
            self.data = self.data[p1[1000:]]
            self.targets = [self.targets[i] for i in p1.tolist()[1000:]]
            #OOD Data
            np.random.seed(3)
            p2 = np.random.permutation(len(self.outpath))
            self.outpath = [self.outpath[i] for i in p2.tolist()[1000:len(p1)]]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if index < len(self.data):
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
        else:
            img_path, target = self.outpath[index - len(self.data)], -1
            img = pil_loader(img_path)
            img = transforms.Resize(32)(img)


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data) + len(self.outpath)