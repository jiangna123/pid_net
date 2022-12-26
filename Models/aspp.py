#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 7/5/21 11:15 AM
# File    : aspp.py
# Describe:
# Author  : jiangna

kinit = 'glorot_normal'

from keras.layers import Conv2D, BatchNormalization, Activation,  Dropout, AveragePooling2D, \
    Concatenate
from keras.engine.topology import Layer, InputSpec
import numpy as np
import tensorflow as tf


def ASPP(inputs, out_channel):
    # 1x1 卷积  rate 1
    # 3x3 卷积  rate 6onfig
    # 3x3 卷积  rate 12
    # 3x3 卷积  rate 18

    """
    感受野的定义： 是卷积神经网络每一层输出的特征图尚像素点在输入图片尚的映射区域大小，特征图上的一个点对应输入图上的区域。
                公式 RFi =(RF(i+1) -1)* stride + Ksize
                    RFi是第i层卷基层的感受野，
                    FRi+1 是（i+1）层上的感受野，
                    stride是卷积步长，
                    Ksize是本层卷积核大小

    atours convolutino(空洞卷积） 增加感受视野，空洞卷积是为了解决FCN思想的语义分割，
    1.输出图像的size要和输入图像size 一致需要upsample
    2.由于FCN中使用pooling 来增大感受野,同时减低分辨率，导致upsample无法还原pooling导致的一些细节信息的损失而提出的，
    为了减小这种损失，自然需要移除pooling层，因此空洞卷积应运而生


    空洞卷积是在标准的卷积中注入了空洞，以此来增加感受野，相比原来的正常卷积，空洞卷积多了个称之为 dilation rate的参数，
    指的是kernel的间隔数量，（一般的卷积dilation_rate=1）


    spp:池化金字塔结构或者叫空间金字塔池化层
    池化层，池化层其实可以理解成一个压缩的过程，无论是AVE还是MAX其实也输入都没啥关系，输出大小直接变为输出一半就完了（参数为2）

    问题出现在全连接层上，假设同一个池化层的输出分别是32*32*1和64*64*1，这就出问题了，
    因为全连接层的权重矩阵W是一个固定值，池化层的不同尺寸的输出会导致全连接层无法进行训练。


    过拟合表现：模型在训练上loss 小，预测准确率高，测试数据loss大，准确率低
    Dropout可以比较有效的缓解过拟合的发生，在一定程度上达到正则化的效果。
         1.随机删除一部分隐藏的神经元，输入输出神经元保持不变
         2.把输入x通过修改的网络前向传播，然后把得到损失结果通过修改网络反向传播，
            一小部分训练样本执行完这个过程后，在没有删除的神经元上安装随机梯度下降法更新对应的参数（w,b)
        3.重复这个过程，
            . 恢复被删掉的神经元（此时被删除的神经元保持原样，而没有被删除的神经元已经有所更新）
            . 从隐藏层神经元中随机选择一个一半大小的子集临时删除掉（备份被删除神经元的参数）。
            . 对一小批训练样本，先前向传播然后反向传播损失并根据随机梯度下降法更新参数（w，b） （没有被删除的那一部分参数得到更新，删除的神经元参数保持被删除前的结果）。

    BatchNormalization
        1. 选择比较大的初始学习率，让你的训练速度飙涨。
        2. BN具有提高网络泛化能力的特性；
        3. 不需要使用局部响应归一化层；
        4. 可以把训练数据彻底打乱

    """

    x1 = Conv2D(out_channel, (1, 1), strides=(1, 1), kernel_initializer=kinit,
                padding="same", dilation_rate=(1, 1))(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(out_channel, (3, 3), strides=(1, 1), kernel_initializer=kinit,
                padding="same", dilation_rate=(6, 6))(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)

    x3 = Conv2D(out_channel, (3, 3), strides=(1, 1), kernel_initializer=kinit,
                padding="same", dilation_rate=(12, 12))(inputs)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)

    x4 = Conv2D(out_channel, (3, 3), strides=(1, 1), kernel_initializer=kinit,
                padding="same", dilation_rate=(18, 18))(inputs)
    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)

    # x5 = AveragePooling2D((1, 1))(inputs)
    # x5 = Conv2D(out_channel, (1, 1), strides=(1, 1), kernel_initializer=kinit, padding="same")(x5)
    # x5 = BatchNormalization()(x5)
    # x5 = Activation('relu')(x5)

    x = Concatenate(axis=3)([x1, x2, x3, x4])

    x = Conv2D(out_channel, (1, 1), strides=(1, 1), kernel_initializer=kinit, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dropout(0.5)(x)

    return x
