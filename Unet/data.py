#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import cv2
import numpy as np
import os
from tensorflow.keras.utils import Sequence
import math

class myDataGenerator(Sequence):
    def __init__(self, img_path, mask_path=None, clsnum=None, batch=32, to_fit=True, shuffle=True, seed=1, n_channels=3, imgsize=(128, 128)):
        self._img_path = img_path
        self._batch = batch
        self._to_fit = to_fit
        np.random.seed(seed)
        self._shuffle = shuffle
        self._n_channels = n_channels
        self._mask_path = mask_path
        self._clsnum = clsnum

        if to_fit:
            assert mask_path is not None, print("must have mask_path if to_fit is True")

        self._data_lst = os.listdir(img_path)

        if isinstance(imgsize, int):
            self._imgsize = (imgsize, imgsize)
        else:
            assert isinstance(imgsize, tuple) and len(imgsize) == 2
            self._imgsize = imgsize

        if self._shuffle:
            np.random.shuffle(self._data_lst)

    def getClsnum(self):
        clsdict = {}
        for imgname in self._data_lst:
            maskfile = os.path.join(self._mask_path, imgname)
            mask = cv2.imread(maskfile, cv2.IMREAD_GRAYSCALE)
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i][j] not in clsdict:
                        clsdict[mask[i][j]] = 0
        self._clsnum = len(clsdict)
        return self._clsnum

    def _getMask(self, batch_file):
        if self._clsnum is None:
            self.getClsnum()
        y = np.zeros((len(batch_file), *self._imgsize, self._clsnum), dtype=np.float)
        for i, imgname in enumerate(batch_file):
            filepath = os.path.join(self._mask_path, imgname)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self._imgsize)
            mask = np.zeros((*self._imgsize, self._clsnum), dtype=np.float)
            for row in range(img.shape[0]):
                for col in range(img.shape[1]):
                    mask[row][col][int(img[row][col])] = 1.0
            y[i] = mask
        return y

    def _getData(self, batch_file):
        X = np.zeros((len(batch_file), *self._imgsize, self._n_channels), dtype=np.float)
        names = []
        for i, imgname in enumerate(batch_file):
            img = None
            filepath = os.path.join(self._img_path, imgname)
            if self._n_channels == 3:
                img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            else:
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img = img / 255.0
            img = cv2.resize(img, self._imgsize)
            X[i] = img
            names.append(imgname)
        return X, names

    def __len__(self):
        return math.ceil(len(self._data_lst) / self._batch)

    def __getitem__(self, index):
        batch_file = self._data_lst[index * self._batch: (index+1) * self._batch]

        x_data, names = self._getData(batch_file)
        if self._to_fit:
            y_data = self._getMask(batch_file)
            return x_data, y_data
        else:
            return x_data, names

    def on_epoch_end(self):
        if self._shuffle:
            np.random.shuffle(self._data_lst)

    def debug_use(self):
        print(self._data_lst[:20])
        print()
        X, y = self.__getitem__(1)
        print("data: ", X.shape)
        print("labels: ", y.shape)

if __name__ == "__main__":
    img_path = '/home/chaoswjz/Documents/PycharmProjects/Unet/train'
    mask_path = '/home/chaoswjz/Documents/PycharmProjects/Unet/trainannot'
    train_data = myDataGenerator(img_path, mask_path, batch=4)
    print(train_data.getClsnum())
    train_data.debug_use()
