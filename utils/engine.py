import torch
import torch.nn as nn
from typing import Iterable
from torch.utils.tensorboard import SummaryWriter
from utils.AverageMeter import AverageMeter

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(model: torch.nn.Module, train_loader: Iterable,
          optimizer: torch.optim.Optimizer, epoch: int, summary: SummaryWriter):
    model.train()
    loss = nn.CrossEntropyLoss()
    train_loss = AverageMeter()
    for step, data in enumerate(train_loader):

        audio, label = data

        label = label.squeeze()
        print(audio.shape)
        pred = model(audio)

        losses = loss(pred, label)
        train_loss.update(losses.item(), audio.size()[0])
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        print("losses : {} , epoch : {}".format(losses, epoch))
    summary.add_scalar('train/loss', train_loss.avg, epoch)

def val(model: torch.nn.Module, val_loader: Iterable, epoch: int, summary: SummaryWriter):
    model.eval()
    val_acc = AverageMeter()
    val_losses = AverageMeter()

    with torch.no_grad():
        loss = nn.CrossEntropyLoss()
        for step, data in enumerate(val_loader):
            img, label = data
            pred = model(img)
            losses = loss(pred, label)
            prec1 = accuracy(pred.data, label)[0]
            val_losses.update(losses.item(), img.size()[0])
            val_acc.update(prec1.item(), img.size()[0])
    summary.add_scalar('val/loss', val_losses.avg, epoch)
    summary.add_scalar('val/acc', val_acc.avg, epoch)
