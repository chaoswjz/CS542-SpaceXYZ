#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.models import load_model
from data import myDataGenerator
import argparse
import cv2
import os
import numpy as np

parser = argparse.ArgumentParser(description="input user customed parameters and file paths")
parser.add_argument("-f", "--file", type=str, required=True, help="input saved model, string type")
parser.add_argument("-p", "--path", type=str, required=True, help="input test images path, string type")
parser.add_argument("-s", "--size", type=tuple, default=(512, 512),
                    help="input image size, should be compatible with the training images sizes, default 512 x 512, tuple type")
parser.add_argument("-b", "--batch", type=int, default=1,
                    help="input batch size, default is 1, int type")
args = parser.parse_args()

if not os.path.exists('./predictions'):
    os.mkdir('./predictions')

model = None
try:
    model = load_model(args.file)
except Exception as e:
    print(e)
    print("cannot load model, please check if models folder exists, if not, please run train.py first")
    exit()

# expected 6 color used as we only have 6 classes
color = {
    0: [0, 0, 255],           # color red for class 0
    1: [255, 0, 0],           # color blue for class 1
    2: [0, 255, 0],           # color green for class 2
    3: [205, 250, 255],       # color lemon chiffon for class 3
    4: [147, 20, 255],        # color pink for class 4
    5: [211, 0, 148],         # color violet for class 5
    6: [192, 192, 192],       # color silver for class 6
    7: [0, 255, 255],         # color yellow for class 7
    8: [255, 255, 0],         # color cyan for class 8
    9: [255, 0, 255],         # color magenta for class 9
    10: [0, 165, 255],        # color orange for class 10
    11: [30, 105, 210],       # color chocolate for class 11
}

def predict(imgs, names):
    preds = model(imgs)
    for i, img in enumerate(imgs):
        predimg = np.zeros((preds.shape[1], preds.shape[2], 3), dtype=np.uint8)
        for r in range(predimg.shape[0]):
            for c in range(predimg.shape[1]):
                predimg[r][c] = color[np.argmax(preds[i][r][c][:])]
        path = os.path.join('./predictions', names[i])
        cv2.imwrite(path, predimg)
        print("  finished {} image(s) in batch".format(i+1))

test_data = myDataGenerator(args.path, to_fit=False, imgsize=args.size, batch=args.batch, shuffle=False)

cnt = 0
for imgs, names in test_data:
    cnt += 1
    template = "processing batch {}, {} imgs..."
    print(template.format(cnt, len(names)))
    predict(imgs, names)
    print("complete predicting batch {}".format(cnt))

print("finished predictions, exit")
