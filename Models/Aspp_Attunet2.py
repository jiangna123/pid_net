#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 7/5/21 11:22 AM
# File    : Aspp_Attunet2.py
# Describe: aspp-se-attunet
# Author  : jiangna
from Models.PidiNet import PidiNet
from Models.aspp import ASPP

from keras.models import *
from keras.layers import *


def squeeze_excitation(input, num_channel, reduction_ratio):

    pool = GlobalMaxPooling2D()(input)
    squeeze = Dense(int(num_channel // reduction_ratio), activation='relu')(pool)
    excitation = Dense(num_channel, activation='sigmoid')(squeeze)
    scale = Multiply()([input, excitation])

    return scale


def Attention_block(input1, input2, filters):
    """
    注意力模块代码
    :param input1: up_conv
    :param input2: conv_blockr
    :param filters:
    :return:
    """
    g1 = Conv2D(filters, kernel_size=1, strides=1, padding='same')(input1)
    g1 = BatchNormalization()(g1)
    x1 = Conv2D(filters, kernel_size=1, strides=1, padding='same')(input2)
    x1 = BatchNormalization()(x1)

    psi = Activation('relu')(add([g1, x1]))
    psi = Conv2D(filters, kernel_size=1, strides=1, padding='same')(psi)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)
    out = multiply([input2, psi])
    return out


def up_conv(input, filters):
    out = UpSampling2D()(input)
    out = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out


def conv_block(input, filters):
    out = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out


def ASPP_AttUNet2(n_classes=2, channels=1, input_height=256, input_width=256):
    input_size = (input_width, input_height, channels)
    inputs = Input(shape=input_size)
    n1 = 64
    filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

    e1 = conv_block(inputs, filters[0])
    e2 = MaxPooling2D(strides=2)(e1)

    e2aspp = ASPP(e2, filters[1])
    e2SE = squeeze_excitation(e2, int(e2.shape[-1]), 16)
    e2 = conv_block(e2, filters[1])
    e2 = Concatenate()([e2, e2aspp])
    # e2 = Concatenate()([e2, e2aspp, eh2SE])
    e2 = conv_block(e2, filters[1])

    e3 = MaxPooling2D(strides=2)(e2)
    e3aspp = ASPP(e3, filters[2])
    e3SE = squeeze_excitation(e3, int(e3.shape[-1]), 16)
    e3 = conv_block(e3, filters[2])
    e3 = Concatenate()([e3, e3aspp])
    # e3 = Concatenate()([e3, e3aspp, e3SE])
    e3 = conv_block(e3, filters[2])

    e4 = MaxPooling2D(strides=2)(e3)
    e4aspp = ASPP(e4, filters[3])
    e4SE = squeeze_excitation(e4, int(e4.shape[-1]), 16)
    e4 = conv_block(e4, filters[3])
    e4 = Concatenate()([e4, e4aspp])
    # e4 = Concatenate()([e4, e4aspp, e4SE])
    e4 = conv_block(e4, filters[3])

    e5 = MaxPooling2D(strides=2)(e4)
    e5aspp = ASPP(e5, filters[4])
    e5SE = squeeze_excitation(e5, int(e5.shape[-1]), 16)
    e5 = conv_block(e5, filters[4])
    e5 = Concatenate()([e5, e5aspp])
    # e5 = Concatenate()([e5, e5aspp, e5SE])
    e5 = conv_block(e5, filters[4])

    d5 = up_conv(e5, filters[3])
    x4 = Attention_block(d5, e4, filters[3])
    d5 = Concatenate()([x4, d5])
    d5 = conv_block(d5, filters[3])

    d4 = up_conv(d5, filters[2])
    x3 = Attention_block(d4, e3, filters[2])
    d4 = Concatenate()([x3, d4])
    d4 = conv_block(d4, filters[2])

    d3 = up_conv(d4, filters[1])
    x2 = Attention_block(d3, e2, filters[1])
    d3 = Concatenate()([x2, d3])
    d3 = conv_block(d3, filters[1])

    d2 = up_conv(d3, filters[0])
    x1 = Attention_block(d2, e1, filters[0])
    d2 = Concatenate()([x1, d2])
    d2 = conv_block(d2, filters[0])

    out = Conv2D(64, 2, padding='same', activation='sigmoid')(d2)
    out = Conv2D(2, 2, padding='same')(out)
    net_out = Conv2D(1, 1, padding='same', activation='sigmoid')(out)  # 添加

    model = Model(input=inputs, output=net_out)
    return model
