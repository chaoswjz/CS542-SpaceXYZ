#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 20:41:55 2018

@author: rz2333
"""
import sys
sys.path.append('/home/rz2333/study/cs524_project/segresnet_solid')

from argparse import ArgumentParser
import os
import random
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


import PIL
from PIL import Image
import numpy as np
from math import ceil

pretrained_net = FeatureResNet()
pretrained_net.load_state_dict(models.resnet34(pretrained=True).state_dict())
net = SegResNet(num_classes, pretrained_net).cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  net = nn.DataParallel(net)

net.to(device)
net.load_state_dict(torch.load('/home/rz2333/Desktop/only_one_style/hollow_wall/crop/119_net.pth'))

def model_usage(path, return_rgb=False, resize=512, crop=None, random_crop=True):
    input = Image.open(path).convert("RGB")
    input_list = []
    pred_list = []
    
    if resize is not None:
        input = input.resize((resize, resize))
        input_list.append(input)
        
    if (crop is not None) and random_crop:
        w, h = input.size
        w_2, h_2 = math.floor(w/2), math.floor(h/2)

        #x1, y1 = random.randint(0, w - self.crop), random.randint(0, h - self.crop)
        x1, y1 = w_2+random.choice([-1,1])*crop, h_2+random.choice([-1,1])*crop
        input = input.crop((min(x1,w_2), min(y1,h_2), max(x1,w_2), max(y1,h_2)))
        input_list.append(input)
        
    if (crop is not None) and (not random_crop):
        w, h = input.size
        w_num, h_num = ceil(w/crop), ceil(h/crop)
        for i in range(w_num):
            for j in range(h_num):
                input_crop = input.crop((i*crop, j*crop, (i+1)*crop, (j+1)*crop))
                input_list.append(input_crop)
        
    input_image = input
    for input_img in input_list:
        w, h = input_img.size
        input_img = torch.ByteTensor(torch.ByteStorage.from_buffer(input_img.tobytes())).view(h, w, 3).permute(2, 0, 1).float().div(255)
        input_img[0].add_(-0.485).div_(0.229)
        input_img[1].add_(-0.456).div_(0.224)
        input_img[2].add_(-0.406).div_(0.225)
        input_img = input_img.contiguous().view(1,3,512,512)
        input_img = Variable(input_img.cuda())
        output = F.log_softmax(net(input_img))
        pred = output.permute(0, 2, 3, 1).contiguous().view(-1, num_classes).max(1)[1].view(1, h, w)
        pred_list.append(pred)
    
    if return_rgb:
        pred_color = []
        for pred in pred_list:
            pred_image = pred.cpu().numpy().reshape((512,512))
            pred_colour = np.zeros((512,512,3))
            pred_colour[pred_image==1,] = [128, 64, 128] 
            pred_colour[pred_image==0,] = [53, 250, 79]
            pred_colour[pred_image==2,] = [172, 178, 173]
            pred_color.append(pred_colour.astype('uint8'))
    
    w, h = input_image.size
    pred_image = np.zeros((w_num*crop,h_num*crop,3))
    order = 0
    for i in range(w_num):
        for j in range(h_num):
            #pred_image[i*crop:(i+1)*crop, j*crop:(j+1)*crop, :] = pred_color[order]
            pred_image[j*crop:(j+1)*crop, i*crop:(i+1)*crop, :] = pred_color[order]
            order += 1
            
    return input_image, pred_list, pred_image.astype('uint8')

test_img = '/home/rz2333/study/cs524_project/spaceXYZ_data/oxygen_png/lot_plan_010216.png'
#test_img = '/home/rz2333/study/cs524_project/data_for_segment/val/img/IIa_SL0801.png'
test_input, _, test_output = model_usage(path=test_img, 
                          return_rgb=True, resize=None, crop=512, random_crop=False)


%matplotlib inline  
plt.imshow(test_output)
plt.imshow(test_input)