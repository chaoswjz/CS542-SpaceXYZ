#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 23:25:05 2018

@author: rz2333
"""
import sys
sys.path.append('/home/rz2333/study/cs524_project/segresnet_solid')

from argparse import ArgumentParser
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.utils import save_image

from data import CityscapesDataset, num_classes, full_to_colour, train_to_full
from model import FeatureResNet, SegResNet


# Setup
parser = ArgumentParser(description='Semantic segmentation')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--workers', type=int, default=4, help='Data loader workers')
parser.add_argument('--epochs', type=int, default=150, help='Training epochs')
parser.add_argument('--crop-size', type=int, default=512, help='Training crop size')
parser.add_argument('--lr', type=float, default=8e-5, help='Learning rate')
parser.add_argument('--momentum', type=float, default=0, help='Momentum')
parser.add_argument('--weight-decay', type=float, default=2e-4, help='Weight decay')
parser.add_argument('--batch-size', type=int, default=10, help='Batch size')
args = parser.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)
if not os.path.exists('results'):
  os.makedirs('results')
plt.switch_backend('agg')  # Allow plotting when running remotely

# Data
train_dataset = CityscapesDataset(split='train', crop=None, flip=True, resize=512, drawcrop=False)
val_dataset = CityscapesDataset(split='val', crop=None, resize=512)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)#, pin_memory=F)
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0)

# Training/Testing
pretrained_net = FeatureResNet()
pretrained_net.load_state_dict(models.resnet34(pretrained=True).state_dict())
net = SegResNet(num_classes, pretrained_net).cuda()
#weights = [0.2, 5, 0.2]
#weights = [5]*512
#class_weights = torch.FloatTensor(weights).cuda()
#crit = nn.BCELoss().cuda()
#crit = nn.BCEWithLogitsLoss(pos_weight=class_weights).cuda()  #
crit = nn.BCEWithLogitsLoss().cuda()
#crit = F.cross_entropy(weight=class_weights)

# Multiple gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  net = nn.DataParallel(net)

net.to(device)

# Construct optimiser
params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
  if 'bn' in key:
    # No weight decay on batch norm
    params += [{'params': [value], 'weight_decay': 0}]
  elif '.bias' in key:
    # No weight decay plus double learning rate on biases
    params += [{'params': [value], 'lr': 2 * args.lr, 'weight_decay': 0}]
  else:
    params += [{'params': [value]}]
optimiser = optim.RMSprop(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
#optimiser = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
scores, mean_scores, mean_loss = [], [], []


def train(e):
  net.train()
  total_loss = []
  
  for i, (input, target, _) in enumerate(train_loader):
    optimiser.zero_grad()
    #input, target = Variable(input.cuda(async=True)), Variable(target.cuda(async=True))
    input, target = Variable(input.cuda()), Variable(target.cuda())
    #output = F.sigmoid(net(input))
    #print('Type of the target is', type(target))
    output = net(input)
    loss = crit(output, target)
    total_loss.append(loss.data[0])

    
    #loss = F.cross_entropy(output, target)
    print(e, i, loss.data[0])
    loss.backward()
    optimiser.step()
    
    
  total_loss = [x.cpu().detach().numpy() for x in total_loss]
  mean_loss.append(np.mean(total_loss))
  es_l = list(range(len(mean_loss)))
  plt.plot(es_l, mean_loss, 'b-')
  plt.xlabel('Epoch')
  plt.ylabel('Mean Loss')
  plt.savefig(os.path.join('/home/rz2333/study/cs524_project/segresnet_solid/results', 'loss.png'))
  plt.close()

# Calculates class intersections over unions
def iou(pred, target):
  ious = []
  # Ignore IoU for background class
  for cls in range(num_classes - 1):
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum().data.cpu().numpy()  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu().numpy() + target_inds.long().sum().data.cpu().numpy() - intersection
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(intersection / max(union, 1))
  return ious


def test(e):
  net.eval()
  total_ious = []
  print('Now start testing')
  for i, (input, _, target) in enumerate(val_loader):
    #print('Testing image '+str(i)+'.')
    #input, target = Variable(input.cuda(async=True), volatile=True), Variable(target.cuda(async=True), volatile=True)
    input, target = Variable(input.cuda()), Variable(target.cuda())
    output = F.log_softmax(net(input))
    b, _, h, w = output.size()
    pred = output.permute(0, 2, 3, 1).contiguous().view(-1, num_classes).max(1)[1].view(b, h, w)
    total_ious.append(iou(pred, target))
    
    
    # Save images
    if i % 1 == 0:
      pred = pred.data.cpu()
      pred_remapped = pred.clone()
      # Convert to full labels
      for k, v in train_to_full.items():
        pred_remapped[pred == k] = v
      # Convert to colour image
      pred = pred_remapped.view(b, 1, h, w)
      pred_colour = torch.zeros(b, 3, h, w)
      for k, v in full_to_colour.items():
        pred_r = torch.zeros(b, 1, h, w)
        pred_r[(pred == k)] = v[0]
        pred_g = torch.zeros(b, 1, h, w)
        pred_g[(pred == k)] = v[1]
        pred_b = torch.zeros(b, 1, h, w)
        pred_b[(pred == k)] = v[2]
        pred_colour.add_(torch.cat((pred_r, pred_g, pred_b), 1))
      save_image(pred_colour[0].float().div(255), os.path.join('results', str(e) + '_' + str(i) + '.png'))
      #save_image(pred_colour[0], os.path.join('/home/rz2333/study/cs524_project/segresnet/results', str(e) + '_' + str(i) + '.png'))

  # Calculate average IoU
  total_ious = torch.Tensor(total_ious).transpose(0, 1)
  ious = torch.Tensor(num_classes - 1)
  for i, class_iou in enumerate(total_ious):
    ious[i] = class_iou[class_iou == class_iou].mean()  # Calculate mean, ignoring NaNs
  print(ious, ious.mean())
  scores.append(ious)

  # Save weights and scores
  torch.save(net.state_dict(), os.path.join('/home/rz2333/study/cs524_project/segresnet_solid/results', str(e) + '_net.pth'))
  torch.save(scores, os.path.join('/home/rz2333/study/cs524_project/segresnet_solid/results', 'scores.pth'))

  # Plot scores
  mean_scores.append(ious.mean())
  es = list(range(len(mean_scores)))
  plt.plot(es, mean_scores, 'b-')
  plt.xlabel('Epoch')
  plt.ylabel('Mean IoU')
  plt.savefig(os.path.join('/home/rz2333/study/cs524_project/segresnet_solid/results', 'ious.png'))
  plt.close()

#test(0)
for e in range(1, args.epochs + 1):
  train(e)
  test(e)
