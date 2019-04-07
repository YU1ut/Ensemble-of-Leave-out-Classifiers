import os
import shutil
import time
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms

import models
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


class Solver(object):
    def __init__(self, args):
        torch.backends.cudnn.benchmark = True
        self.args = args
        self.best_prec1 = 0
        if not os.path.isdir(self.args.checkpoint):
            mkdir_p(self.args.checkpoint)

        self.set_dataloater()

        self.set_model()

        self.set_optimizer()

        self.set_criterion()

        title = 'Proposed'
        if self.args.resume:
            print('==> Resuming from checkpoint..')
            assert os.path.isfile(self.args.resume), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(self.args.resume)
            self.args.start_epoch = checkpoint['epoch']
            self.best_prec1 = checkpoint['best_prec1']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.opt.load_state_dict(checkpoint['opt'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(self.args.resume, checkpoint['epoch']))
            self.logger = Logger(os.path.join(self.args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            self.logger = Logger(os.path.join(self.args.checkpoint, 'log.txt'), title=title)
            self.logger.set_names(['Learning Rate', 'Train Loss', 'Valid Acc.'])

    def set_dataloater(self):
        raise NotImplementedError

    def set_model(self):
        self.model = models.DenseNet(self.args.num_classes).cuda()
        

    def set_optimizer(self):
        self.opt = torch.optim.SGD(self.model.parameters(), self.args.lr,
                                        momentum=self.args.momentum,
                                        weight_decay=self.args.weight_decay)

    def set_criterion(self):
        self.criterion_cel = nn.CrossEntropyLoss().cuda()

    def train(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        self.model.train()

        ood_train_iter = iter(self.ood_train_loader)
        end = time.time()
        bar = Bar('Training', max=len(self.id_train_loader))
        for batch_idx, (data_id, target_id) in enumerate(self.id_train_loader):

            try:
                data_ood, _  = ood_train_iter.next()
            except:
                ood_train_iter = iter(self.ood_train_loader)
                data_ood, _ = ood_train_iter.next()

            # measure data loading time
            data_time.update(time.time() - end)

            data_id, target_id = data_id.cuda(), target_id.cuda(non_blocking=True)
            data_ood = data_ood.cuda()

            batch_size_id = len(target_id)
            
            output_id = self.model(data_id)

            E_id = -torch.mean(torch.sum(F.log_softmax(output_id, dim=1) * F.softmax(output_id, dim=1), dim=1))
            
            output_ood = self.model(data_ood)

            E_ood = -torch.mean(torch.sum(F.log_softmax(output_ood, dim=1) * F.softmax(output_ood, dim=1), dim=1))

            loss = self.criterion_cel(output_id, target_id) + self.args.beta * torch.clamp(0.4 + E_id - E_ood, min=0)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            losses.update(loss.item(), batch_size_id)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}'.format(
                        batch=batch_idx + 1,
                        size=len(self.id_train_loader),
                        data=data_time.val,
                        bt=batch_time.val,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        )
            bar.next()
        bar.finish()
        return losses.avg

    def val(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.model.eval()

        gts = []
        probs = []

        end = time.time()
        bar = Bar('Testing ', max=len(self.val_loader))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                # measure data loading time
                data_time.update(time.time() - end)

                data, target = data.cuda(), target.cuda(non_blocking=True)
                output = self.model(data)
                prob = F.softmax(output, dim=1)

                for i in range(len(prob)):
                    gts.append(target[i].item())
                    probs.append(prob[i].cpu().numpy())
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
            
                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                            batch=batch_idx + 1,
                            size=len(self.val_loader),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            )
                bar.next()
            bar.finish()

        diff = [0, 1]
        cifar = []
        other = []
        for i in range(len(gts)):
            gt = gts[i]
            prob = probs[i]
            if gt >= 0:
                cifar.append(np.max(prob))
            else:
                other.append(np.max(prob))
            diff.append(np.max(prob)+10e-5)
        diff = sorted(list(set(diff)))
        cifar, other = np.array(cifar), np.array(other)

        #calculate the AUROC
        aurocBase = 0.0
        fprTemp = 1.0
        for delta in diff:
            tpr = np.sum(np.sum(cifar >= delta)) / np.float(len(cifar))
            fpr = np.sum(np.sum(other >= delta)) / np.float(len(other))
            aurocBase += (-fpr+fprTemp)*tpr
            fprTemp = fpr
        aurocBase += fpr * tpr

        print (f"Val AUROC: {aurocBase} ")
        auroc = aurocBase
        is_best = auroc > self.best_prec1
        self.best_prec1 = max(auroc, self.best_prec1)

        self.save_checkpoint({
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'best_prec1': self.best_prec1,
            'opt' : self.opt.state_dict(),
        }, is_best, checkpoint=self.args.checkpoint)
        return auroc

    def save_checkpoint(self, state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(checkpoint, filename)
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

    def adjust_learning_rate(self, epoch):
        if epoch in self.args.schedule:
            self.args.lr *= self.args.gamma
            for param_group in self.opt.param_groups:
                param_group['lr'] = self.args.lr