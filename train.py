from __future__ import print_function
import argparse
import os
import torch
from torchvision import transforms
from torchvision import datasets
from eloc_solver import Solver
from utils import savefig, CIFAR10Mix, CIFAR100Mix
import models
import numpy as np
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                    help='number of known classes (default: 10)')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--schedule', type=int, nargs='+', default=[50, 75],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('-c', '--checkpoint', default='cifar10_out_checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: pretrain_checkpoint)')
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--wide', dest='wide', action='store_true',
                    help='use wide-resnet')
parser.add_argument('--in-dataset', default="cifar10", type=str,
                    help='training set')

parser.add_argument('--beta', type=float, default=0.2, metavar='TH',
                    help='weight on entropy loss (default: 0.2)')
parser.add_argument('--fold', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 90)')
args = parser.parse_args()

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()


class MySolver(Solver):
    def __init__(self, args):
        super(MySolver, self).__init__(args)

    def set_model(self):
        if args.wide:
            self.model = models.WideResNet(int(args.num_classes*4/5), dropRate=0.3).cuda()
        else:
            self.model = models.DenseNet(int(args.num_classes*4/5)).cuda()

    def set_dataloater(self):
        if args.in_dataset == 'cifar10':
            id_train_dataset = datasets.CIFAR10('data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
                    ]))

            ood_train_dataset = datasets.CIFAR10('data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
                    ]))

            val_dataset = CIFAR10Mix('data', './data/iSUN', train=False, val=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
                            ]))
        elif args.in_dataset == 'cifar100':
            args.num_classes = 100
            id_train_dataset = datasets.CIFAR100('data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
                    ]))

            ood_train_dataset = datasets.CIFAR100('data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
                    ]))

            val_dataset = CIFAR100Mix('data', './data/iSUN', train=False, val=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
                            ]))

        id_train_dataset, ood_train_dataset = split_dataset(id_train_dataset, ood_train_dataset)

        self.id_train_loader = torch.utils.data.DataLoader(id_train_dataset, 
            batch_size=args.batch_size, shuffle=True, num_workers=4)

        self.ood_train_loader = torch.utils.data.DataLoader(ood_train_dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=4)

        self.val_loader = torch.utils.data.DataLoader(val_dataset,
            batch_size=args.batch_size, shuffle=False, num_workers=4)

def split_dataset(id_train_dataset, ood_train_dataset):
    num_classes = args.num_classes
    np.random.seed(3)
    p1 = np.random.permutation(num_classes).tolist()
    nclass_split = int(num_classes/5)
    Out_classes = p1[(args.fold - 1) * nclass_split : nclass_split * args.fold]
    In_classes = []
    for item in p1:
        if item not in Out_classes:
            In_classes.append(item)
    print (f"# ID classes : {len(In_classes)} # OOD classes: {len(Out_classes)}")
    indata = []
    inlabel = []
    outdata = []
    outlabel = []
    for i in range(len(id_train_dataset.train_data)):
        if int(id_train_dataset.train_labels[i]) in In_classes:
            indata.append(id_train_dataset.train_data[i])
            inlabel.append(In_classes.index(id_train_dataset.train_labels[i]))
        else:
            outdata.append(id_train_dataset.train_data[i])
            outlabel.append(-1)

    id_train_dataset.train_data = indata
    id_train_dataset.train_labels = inlabel

    ood_train_dataset.train_data = outdata
    ood_train_dataset.train_labels = outlabel

    return id_train_dataset, ood_train_dataset

if __name__ == '__main__':
    solver = MySolver(args)
    for epoch in range(args.start_epoch, args.epochs):
        solver.adjust_learning_rate(epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, solver.args.lr))
        # train for one epoch
        train_loss = solver.train(epoch)
        # evaluate on validation set
        val_acc = solver.val(epoch)
        # append logger file
        solver.logger.append([solver.args.lr, train_loss, val_acc])

    solver.logger.close()
    solver.logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(solver.best_prec1)