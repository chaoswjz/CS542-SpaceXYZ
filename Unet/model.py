#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dropout, Conv2DTranspose, MaxPool2D, BatchNormalization, Concatenate, UpSampling2D
from tensorflow.nn import relu, sigmoid, tanh, softmax, conv2d


'''
Customize activation function layer
can perform relu, sigmoid, tanh, softmax
default return relu activation
mode should be 'relu', 'sigmoid', 'tanh', softmax', case insensitive
'''
class myActivation(Layer):
    def __init__(self, mode='relu'):
        super(myActivation, self).__init__()
        mode = mode.lower()
        assert mode in ['relu', 'sigmoid', 'tanh', 'softmax']
        self._mode = mode

    def call(self, x):
        if self._mode == 'sigmoid':
            return sigmoid(x)
        elif self._mode == 'tanh':
            return tanh(x)
        elif self._mode == 'softmax':
            return softmax(x)
        else:
            return relu(x)


class myConv2d(Layer):
    def __init__(self, filters, kernel_size=[3, 3], stride=[1, 1, 1, 1], padding='same', activation='relu', batchnorm=True):
        super(myConv2d, self).__init__()

        self.bnlayer = BatchNormalization()
        self._batchnorm = batchnorm

        self._filters = filters

        self._kernel_size = None
        if isinstance(kernel_size, int):
            self._kernel_size = [kernel_size, kernel_size]
        else:
            assert len(kernel_size) == 2
            assert isinstance(kernel_size, list) or isinstance(kernel_size, tuple)
            self._kernel_size = kernel_size

        self._stride = None
        if isinstance(stride, int):
            self._stride = [1, stride, stride, 1]
        elif isinstance(stride, list) and len(stride) == 2:
            self._stride = [1, stride[0], stride[1], 1]
        elif isinstance(stride, tuple) and len(stride) == 2:
            self._stride = [1, stride[0], stride[1], 1]
        else:
            assert len(stride) == 4
            assert isinstance(stride, list) or isinstance(stride, tuple)
            self._stride = stride

        assert isinstance(padding, str) and (padding.lower() in ['valid', 'same'])
        self._padding = padding.upper()

        assert isinstance(activation, str) and (activation.lower() in ['relu', 'sigmoid', 'tanh', 'softmax'])
        self._activation = myActivation(activation.lower())

        self._batchnorm = batchnorm

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                      shape=[self._kernel_size[0], self._kernel_size[1], int(input_shape[-1]), self._filters])
        self.bias = self.add_weight("bias",
                                    shape=[self._filters])

    def call(self, x, training=False):
        convout = tf.nn.conv2d(x, self.kernel, self._stride, self._padding)
        convbias = tf.add(convout, self.bias)
        if self._batchnorm:
            convbias = self.bnlayer(convbias, training=training)
        return self._activation(convbias)


class contractBlock(Layer):
    def __init__(self, filters, droprate=0.5, kernel=3,
                 stride=1, padding='same', activation='relu', batchnorm=True):
        super(contractBlock, self).__init__()

        self.conv1 = myConv2d(filters, kernel_size=kernel, stride=stride, padding=padding,
                              activation=activation, batchnorm=batchnorm)
        self.conv2 = myConv2d(filters, kernel_size=kernel, stride=stride, padding=padding,
                              activation=activation, batchnorm=batchnorm)
        self.pool = MaxPool2D()
        self.drop = Dropout(droprate)

    def call(self, x, training=False):
        c = self.conv1(x, training=training)
        c = self.conv2(c, training=training)
        p = self.pool(c)
        p = self.drop(p, training=training)

        return c, p


class bottleNeck(Layer):
    def __init__(self, filters, kernel=3, stride=1,
                 padding='same', activation='relu', batchnorm=True):
        super(bottleNeck, self).__init__()
        self.conv1 = myConv2d(filters, kernel_size=kernel, stride=stride, padding=padding,
                              activation=activation, batchnorm=batchnorm)
        self.conv2 = myConv2d(filters, kernel_size=kernel, stride=stride, padding=padding,
                              activation=activation, batchnorm=batchnorm)

    def call(self, x, training=False):
        c = self.conv1(x, training=training)
        c = self.conv2(c, training=training)

        return c


class expandBlock(Layer):
    def __init__(self, filters, droprate=0.5, convkernel=3, transkernel=2,
                 convstride=1, transstride=(2, 2), convpadding='same',
                 transpadding='same', activation='relu', batchnorm=True):
        super(expandBlock, self).__init__()

        self.convtrans = Conv2DTranspose(filters, transkernel, transstride, transpadding)
        #self.upsample = UpSampling2D()
        #self.conv = myConv2d(filters, kernel_size=convkernel, stride=convstride, padding=convpadding,
        #                      activation=activation, batchnorm=batchnorm)
        self.concat = Concatenate()
        self.drop = Dropout(droprate)

        self.conv1 = myConv2d(filters, kernel_size=convkernel, stride=convstride, padding=convpadding,
                              activation=activation, batchnorm=batchnorm)
        self.conv2 = myConv2d(filters, kernel_size=convkernel, stride=convstride, padding=convpadding,
                              activation=activation, batchnorm=batchnorm)

    def call(self, x, training=False):
        assert isinstance(x, list) or isinstance(x, tuple)
        assert len(x) == 2
        copy, up = x
        trans = self.convtrans(up)
        #upsamp = self.upsample(up)
        #trans = self.conv(upsamp)
        concat = self.concat([copy, trans])
        out = self.drop(concat, training=training)
        out = self.conv1(out, training=training)
        out = self.conv2(out, training=training)

        return out


class myUnet(Model):
    def __init__(self, numcls, filter_size=[64, 128, 256, 512, 1024], droprate=0.5,
                 convkernel=3, transkernel=2, convstride=1, transstride=(2, 2),
                 convpadding='same', transpadding='same', activation='relu', batchnorm=True):
        super(myUnet, self).__init__()
        assert isinstance(filter_size, list) or isinstance(filter_size, tuple), print("filter_size must be tuple or list")
        assert len(filter_size) == 5, print("filter_size length must be 5")

        self.down1 = contractBlock(filter_size[0], droprate, convkernel, convstride, convpadding, activation, batchnorm)
        self.down2 = contractBlock(filter_size[1], droprate, convkernel, convstride, convpadding, activation, batchnorm)
        self.down3 = contractBlock(filter_size[2], droprate, convkernel, convstride, convpadding, activation, batchnorm)
        self.down4 = contractBlock(filter_size[3], droprate, convkernel, convstride, convpadding, activation, batchnorm)

        self.bottom = bottleNeck(filter_size[4], convkernel, convstride, convpadding, activation, batchnorm)

        self.up4 = expandBlock(filter_size[3], droprate, convkernel, transkernel, convstride, transstride,
                               convpadding, transpadding, activation, batchnorm)
        self.up3 = expandBlock(filter_size[2], droprate, convkernel, transkernel, convstride, transstride,
                               convpadding, transpadding, activation, batchnorm)
        self.up2 = expandBlock(filter_size[1], droprate, convkernel, transkernel, convstride, transstride,
                               convpadding, transpadding, activation, batchnorm)
        self.up1 = expandBlock(filter_size[0], droprate, convkernel, transkernel, convstride, transstride,
                               convpadding, transpadding, activation, batchnorm)

        self.final = myConv2d(numcls, 1, activation='softmax')

    def call(self, x, training=False):
        c1, d1 = self.down1(x, training=training)
        c2, d2 = self.down2(d1, training=training)
        c3, d3 = self.down3(d2, training=training)
        c4, d4 = self.down4(d3, training=training)

        b = self.bottom(d4, training=training)

        o4 = self.up4([c4, b], training=training)
        o3 = self.up3([c3, o4], training=training)
        o2 = self.up2([c2, o3], training=training)
        o1 = self.up1([c1, o2], training=training)

        out = self.final(o1, training=training)

        return out
