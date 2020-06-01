#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, Adadelta, SGD, RMSprop
from tensorflow.keras.metrics import Mean, CategoricalAccuracy
from model import myUnet
from data import myDataGenerator
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys

parser = argparse.ArgumentParser(description='process user customed parameters and paths')
parser.add_argument("-ti", "--timgs", help="input path of the training imgs folder, string type", type=str, required=True)
parser.add_argument("-tm", "--tmasks", help="input path of the training masks foler, string type", type=str, required=True)
parser.add_argument("-vi", "--vimgs", help="input path of the validation imgs folder, string type", type=str, required=True)
parser.add_argument("-vm", "--vmasks", help="input path of the validation masks foler, string type", type=str, required=True)
parser.add_argument("-t", "--train", default=False, type=bool, help="training including validation data if True, default is false, bool type, validation metrics are no longer accurate as it's also part of the training data")
parser.add_argument("-e", "--epochs", help="input epochs of training, default is 5, integer type", type=int, default=5)
parser.add_argument("-s", "--size", type=tuple, default=(512, 512),
                    help="input the imgs target size, tuple type, default is 512 x 512")
parser.add_argument("-ck", "--convkernel", default=3,
                    help="input the convlution kernel size, can be 1 interge, tuple/list of 2/4 integers, default is 3")
parser.add_argument("-tk", "--transkernel", default=2,
                    help="input the transpose convolution kernel size, can be 1 interge, tuple/list of 2/4 integers, default is 2")
parser.add_argument("-cs", "--convstride", default=1,
                    help="input convolution stride, can be a integer, tuple/list of 2/4 integers, default is 1")
parser.add_argument("-ts", "--transstride", default=2,
                    help="input transpose convolution stride, can be a integer, tuple/list of 2/4 integers, default is 2")
parser.add_argument("-cp", "--convpadding", default='same', type=str,
                    help="input convolution padding type, string type, default is 'same'")
parser.add_argument("-tp", "--transpadding", default='same', type=str,
                    help="input transepose padding type, string type, default is 'same'")
parser.add_argument("-a", "--activation", default='relu', type=str,
                    help="input activation type for convolutions except for the last one, default is 'relu', string type")
parser.add_argument("-d", "--droprate", default=0.5, type=float,
                    help="input dropout rate, default is 0.5, float type")
parser.add_argument("-f", "--filters", default=[64, 128, 256, 512, 1024],
                    help="input filters, list/tuple type of 5 integers, default is 64, 128, 256, 512, 1024")
parser.add_argument("-bn", "--batchnorm", default=True, type=bool,
                    help="Turn on/off batch normalization, bool type, default is True")
parser.add_argument("-b", "--batch", type=int, default=1,
                    help="input batch size for training/validation, integer type, default is 1")
parser.add_argument("-o", "--optimizer", default='Adam', type=str,
                    help="select optimizer from Adam, RMSprop, Adadelta, SGD, string type, default is 'Adam'")
parser.add_argument("-lr", "--learningrate", type=float, default=0.001,
                    help="input learning rate, float type, default is 0.001")
parser.add_argument("-c", "--checkpoint", default='', type=str,
                    help="input the checkpoint path to continue training, default is empty to train from scratch, string type")

args = parser.parse_args()

train_data = myDataGenerator(args.timgs, args.tmasks, batch=args.batch, imgsize=args.size)
numcls = train_data.getClsnum()
valid_data = myDataGenerator(args.vimgs, args.vmasks, numcls, batch=args.batch, imgsize=args.size)

model = myUnet(numcls, args.filters, args.droprate, args.convkernel, args.transkernel, args.convstride,
               args.transstride, args.convpadding, args.transpadding, args.activation, args.batchnorm)

if args.checkpoint != '':
    try:
        model.load(args.checkpoint)
    except Exception as e:
        print(e)
        exit()

lossobj = CategoricalCrossentropy()

optimizer = None

if args.optimizer.lower() == 'adadelta':
    optimizer = Adadelta(args.learningrate)
elif args.optimizer.lower() == 'sgd':
    optimizer = SGD(args.learningrate)
elif args.optimizer.lower() == 'rmsprop':
    optimizer = RMSprop(args.learningrate)
else:
    optimizer = Adam(args.learningrate)

train_loss = Mean(name='train_loss')
train_accuracy = CategoricalAccuracy(name='train_accuracy')

valid_loss = Mean(name='valid_loss')
valid_accuracy = CategoricalAccuracy(name='valid_accuracy')

@tf.function
def trainStep(imgs, masks):
    with tf.GradientTape() as tape:
        preds = model(imgs, training=True)
        loss = lossobj(masks, preds)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
    train_accuracy(masks, preds)

@tf.function
def validStep(imgs, masks):
    preds = model(imgs)
    loss = lossobj(masks, preds)

    valid_loss(loss)
    valid_accuracy(masks, preds)

tloss = []
vloss = []
taccuracy = []
vaccuracy = []
bestloss = sys.maxsize

for epoch in range(args.epochs):
    train_loss.reset_states()
    train_accuracy.reset_states()

    if not args.train:
        valid_loss.reset_states()
        valid_accuracy.reset_states()

    for imgs, masks in train_data:
        trainStep(imgs, masks)

    for imgs, masks in valid_data:
        if not args.train:
            validStep(imgs, masks)
        else:
            trainStep(imgs, masks)

    template1 = "epoch[{0}/{1}] training: mean loss: {2}, accuracy: {3}"
    template2 = "validation: mean loss: {0}, accuracy: {1}"
    if not args.train:
        print(template1.format(epoch+1, args.epochs, train_loss.result(), train_accuracy.result()), end='\t')
        print(template2.format(valid_loss.result(), valid_accuracy.result()))
    else:
        print(template1.format(epoch+1, args.epochs, train_loss.result(), train_accuracy.result()))

    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')

    if not args.train:
        if valid_loss.result() < bestloss:
            bestloss = valid_loss.result()
            model.save_weights('./checkpoints/mycheckpoint.h5')
    else:
        if train_loss.result() < bestloss:
            bestloss = train_loss.result()
            model.save_weights('./checkpoints/mycheckpoint.h5')

    tloss.append(train_loss.result())
    taccuracy.append(train_accuracy.result())

    if not args.train:
        vloss.append(valid_loss.result())
        vaccuracy.append(valid_accuracy.result())

model.summary()

print("\nstart saving model ...")
if not os.path.exists('./models'):
    os.mkdir('./models')
model.save('./models/myUnet')
print("model saved")

# draw accuracy graph and loss graph
x = range(1, args.epochs+1)

if not args.train:
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].plot(x, tloss, 'r', x, vloss, 'b')
    axs[0].set_title('loss over epochs')
    axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('loss')
    axs[0].get_xaxis().set_major_locator(MaxNLocator(integer=True))
    fig.suptitle('Training and validation loss and accuracy over epochs', fontsize=16)

    axs[1].plot(x, taccuracy, 'r', x, vaccuracy, 'b')
    axs[1].set_xlabel('epochs')
    axs[1].set_title('accuracy over epochs')
    axs[1].set_ylabel('accuracy')
    axs[1].get_xaxis().set_major_locator(MaxNLocator(integer=True))

    plt.savefig('plot.png')

else:
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].plot(x, tloss, 'r')
    axs[0].set_title('training loss over epochs')
    axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('loss')
    axs[0].get_xaxis().set_major_locator(MaxNLocator(integer=True))
    fig.suptitle('Training loss and accuracy over epochs', fontsize=16)

    axs[1].plot(x, taccuracy, 'r')
    axs[1].set_xlabel('epochs')
    axs[1].set_title('training accuracy over epochs')
    axs[1].set_ylabel('accuracy')
    axs[1].get_xaxis().set_major_locator(MaxNLocator(integer=True))

    plt.savefig('plot.png')
