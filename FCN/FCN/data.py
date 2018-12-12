#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 22:13:13 2018

@author: rz2333
"""

import os
import random
import PIL
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
import math


# Labels: 0 for room, 1 for wall, 3 for opening, 2 for wall_gap, 255 for void
num_classes = 2
#full_to_train = {-1: 19, 0: 19, 1: 19, 2: 19, 3: 19, 4: 19, 5: 19, 6: 19, 7: 0, 8: 1, 9: 19, 10: 19, 11: 2, 12: 3, 13: 4, 14: 19, 15: 19, 16: 19, 17: 5, 18: 19, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 19, 30: 19, 31: 16, 32: 17, 33: 18}
#train_to_full = {0: 7, 1: 8, 2: 11, 3: 12, 4: 13, 5: 17, 6: 19, 7: 20, 8: 21, 9: 22, 10: 23, 11: 24, 12: 25, 13: 26, 14: 27, 15: 28, 16: 31, 17: 32, 18: 33, 19: 0}
#full_to_colour = {0: (0, 0, 0), 7: (128, 64, 128), 8: (244, 35, 232), 11: (70, 70, 70), 12: (102, 102, 156), 13: (190, 153, 153), 17: (153, 153, 153), 19: (250, 170, 30), 20: (220, 220, 0), 21: (107, 142, 35), 22: (152, 251, 152), 23: (70, 130, 180), 24: (220, 20, 60), 25: (255, 0, 0), 26: (0, 0, 142), 27: (0, 0, 70), 28: (0, 60,100), 31: (0, 80, 100), 32: (0, 0, 230), 33: (119, 11, 32)}
full_to_train = {0:1, 255:0}
#full_to_train = {0:1, 50:0, 100:1, 200:1, 255:1}
#train_to_full = {0:0, 1:1, 2:2, 3:3, 4:255}
train_to_full = {0:1, 1:0}
#full_to_colour = {255: (0, 0, 0), 0: (128, 64, 128), 1: (244, 35, 232), 2: (70, 70, 70), 3: (102, 102, 156)}
full_to_colour = {0: (172, 178, 173), 1: (53, 250, 79), 2:(102,156,102), 3:(128, 64, 128)}
#data_dir = '/Users/rz2333/Downloads/Study/BU/Fall_2018/CS542_ml/Final_project/Data/ImagesGT'

class CityscapesDataset(Dataset):
    def __init__(self, split='train', crop=512, flip=False, drawcrop=True, resize=None):
        super().__init__()
        self.crop = crop
        self.flip = flip
        self.inputs = []
        self.targets = []
        self.dir = '/home/rz2333/study/cs524_project/data_spaceXYZ'
        self.drawcrop = drawcrop
        self.resize = resize

        for root, _, filenames in os.walk(os.path.join(self.dir, split)):
            for filename in filenames:
                if (os.path.splitext(filename)[1] in ['.png', '.jpg']) and ('img' in root):
                    #filename_base = '.'.join(os.path.splitext(filename)[0].split('_'))
                    self.inputs.append(os.path.join(root, filename))
                    target_root = os.path.join(self.dir, split, 'label')
                    #filename_base = '.'.join(os.path.splitext(filename)[0].split('_'))
                    filename_base = os.path.splitext(filename)[0]
                    self.targets.append(os.path.join(target_root, filename_base+'_object1.png'))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
    # Load images and perform augmentations with PIL
        input, target = Image.open(self.inputs[i]).convert("RGB"), Image.open(self.targets[i]).convert('L')
    # Random uniform crop
        if self.crop is not None:
            w, h = input.size
            #w_2, h_2 = math.floor(w/2), math.floor(h/2)

            x1, y1 = random.randint(0, w - self.crop), random.randint(0, h - self.crop)
            #x1, y1 = w_2+random.choice([-1,1])*self.crop, h_2+random.choice([-1,1])*self.crop
            #input, target = input.crop((min(x1,w_2), min(y1,h_2), max(x1,w_2),  max(y1,h_2))), target.crop((min(x1,w_2), min(y1,h_2), max(x1,w_2),  max(y1,h_2)))
            input, target = input.crop((x1, y1, x1 + self.crop, y1 + self.crop)), target.crop((x1, y1, x1 + self.crop, y1 + self.crop))
    # Random horizontal flip
        if self.flip:
            if random.random() < 0.5:
                input, target = input.transpose(Image.FLIP_LEFT_RIGHT), target.transpose(Image.FLIP_LEFT_RIGHT)
        if (self.drawcrop) and (i % 1 == 0):
            input.save(os.path.join('/home/rz2333/study/cs524_project/segresnet_solid/results', 'crop_image'+str(i)+'.png'),"png")
            target.save(os.path.join('/home/rz2333/study/cs524_project/segresnet_solid/results', 'crop_target'+str(i)+'.png'),"png")
        if self.resize is not None:
            input,target = input.resize((self.resize, self.resize)), target.resize((self.resize, self.resize))
        #return input, target
    # Convert to tensors
        w, h = input.size
        # Convert grey to RGB
#        if input.mode != 'RGB':
#            input_np = np.stack((np.array(input), np.array(input),np.array(input)), axis=-1)
#            input = torch.ByteTensor(torch.ByteStorage.from_buffer(input_np.tobytes())).view(h, w, 3).permute(2, 0, 1).float().div(255)
#        elif input.mode == 'RGB':
#            input = torch.ByteTensor(torch.ByteStorage.from_buffer(input.tobytes())).view(h, w, 3).permute(2, 0, 1).float().div(255)

        input = torch.ByteTensor(torch.ByteStorage.from_buffer(input.tobytes())).view(h, w, 3).permute(2, 0, 1).float().div(255)
        target = torch.ByteTensor(torch.ByteStorage.from_buffer(target.tobytes())).view(h, w).long()
        # Normalise input
        input[0].add_(-0.485).div_(0.229)
        input[1].add_(-0.456).div_(0.224)
        input[2].add_(-0.406).div_(0.225)
        # Convert to training labels
        remapped_target = target.clone()
        for k, v in full_to_train.items():
            remapped_target[target == k] = v
        # Create one-hot encoding
        target = torch.zeros(num_classes, h, w)
        for c in range(num_classes):
            target[c][remapped_target == c] = 1
        return input, target, remapped_target  # Return x, y (one-hot), y (index)

