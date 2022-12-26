#!/usr/bin/env python
# -*- coding: utf-8 -*-
# file:PidiNet 
# Author  : jiangna
# desctription： 
# datetime： 22-10-29 下午2:47
# ============================

from keras.layers import *


def csam_block(inputs, filter):
    input = Activation('relu')(inputs)

    input = Conv2D(filter, (1, 1), padding='same')(input)
    input = BatchNormalization()(input)
    input = Activation('relu')(input)

    input = Conv2D(4, (3, 3), padding='same')(input)
    input = BatchNormalization()(input)
    input = Activation('relu')(input)

    out = Conv2D(1, 1, padding='same', activation='sigmoid')(input)
    out = Multiply()([inputs, out])

    cv1 = Conv2D(1, 1, padding="same")(out)
    cv1 = BatchNormalization()(cv1)
    out = Activation('relu')(cv1)

    return out


def cdcm_block(inputs, filters):
    input = Activation('relu')(inputs)

    input = Conv2D(filters, 1, padding='same')(input)
    input = BatchNormalization()(input)
    input = Activation('relu')(input)

    # 差分卷积
    x1 = Conv2D(filters, (3, 3), padding="same", dilation_rate=(5, 5))(input)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(filters, (3, 3), padding="same", dilation_rate=(7, 7))(input)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)

    x3 = Conv2D(filters, (3, 3), padding="same", dilation_rate=(9, 9))(input)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)

    x4 = Conv2D(filters, (3, 3), padding="same", dilation_rate=(11, 11))(input)
    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)

    x = Concatenate(axis=3)([x1, x2, x3, x4])
    x = Dropout(0.5)(x)

    return x


def pdc_block(input, filters):
    input = SeparableConv2D(filters, kernel_size=(3, 3), padding="same")(input)
    input = BatchNormalization()(input)
    out = Activation("relu")(input)

    return out


def conv_block(input, filters):
    out = pdc_block(input, filters)

    out = Conv2D(filters, kernel_size=(1, 1), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Concatenate()([input, out])

    return out


def up_conv(input, n):
    out = UpSampling2D(size=1 * n)(input)
    out = Conv2D(1, kernel_size=(3, 3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out


def PidiNet(inputs):
    n1 = 1
    filters = [n1, n1 * 2, n1 * 4, n1 * 8]
    e1 = conv_block(inputs, n1)
    e1 = conv_block(e1, filters[0])
    e1 = conv_block(e1, filters[0])
    x1 = conv_block(e1, filters[0])
    c1 = cdcm_block(x1, filters[0])
    c1 = csam_block(c1, filters[0])
    c1 = up_conv(c1, filters[0])

    e2 = MaxPooling2D(strides=2)(x1)
    e2 = conv_block(e2, filters[1])
    e2 = conv_block(e2, filters[1])
    e2 = conv_block(e2, filters[1])
    x2 = conv_block(e2, filters[1])
    c2 = cdcm_block(x2, filters[1])
    c2 = csam_block(c2, filters[1])
    c2 = up_conv(c2, filters[1])

    e3 = MaxPooling2D(strides=2)(x2)
    e3 = conv_block(e3, filters[2])
    e3 = conv_block(e3, filters[2])
    e3 = conv_block(e3, filters[2])
    x3 = conv_block(e3, filters[2])
    c3 = cdcm_block(x3, filters[2])
    c3 = csam_block(c3, filters[2])
    c3 = up_conv(c3, filters[2])

    e4 = MaxPooling2D(strides=2)(x3)
    e4 = conv_block(e4, filters[3])
    e4 = conv_block(e4, filters[3])
    e4 = conv_block(e4, filters[3])
    x4 = conv_block(e4, filters[3])
    c4 = cdcm_block(x4, filters[3])
    c4 = csam_block(c4, filters[3])
    c4 = up_conv(c4, filters[3])
    # c4 = up_conv(c4, 1, filters[3] * 2)

    out = Concatenate()([c1, c2, c3, c4])
    out = Conv2D(1, 1, padding='same', activation='sigmoid', name='pid_out')(out)

    return out
