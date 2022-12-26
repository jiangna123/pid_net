#!/usr/bin/env python
# -*- coding: utf-8 -*-
# file:xor
# Author  : jiangna
# desctription： 指标计算test
# datetime： 22-12-1 上午9:41
# ============================

import argparse
import cv2
import glob
import os

from keras.preprocessing.image import ImageDataGenerator, array_to_img
import keras.backend as K
import numpy as np

from PIL import Image

#  a,b a+b
#  0,0 0
#  1,0 1
#  0,1 1
#  1,1 0


"""

a = np.array([[1, 0, 1],
              [0, 1, 0],
              [0, 0, 0]])
b = np.array([[1, 1, 1],
              [0, 0, 0],
              [1, 0, 0]])

# c = np.logical_or(a, b)
c = np.maximum(a, b, out=None)
print(c)

"""
parser = argparse.ArgumentParser()
parser.add_argument("--datatest_name", type=str, default="archive_train256_3")  # attunet-mszs
parser.add_argument("--channels", type=int, default=1)  # 默认单通道
parser.add_argument("--as_gray", type=bool, default=False)  # 3 False , 1 True
parser.add_argument("--input_height", type=int, default=256)
parser.add_argument("--input_width", type=int, default=256)

parser.add_argument("--gpu_count", type=int, default=2)
parser.add_argument("--n_classes", type=int, default=2)
args = parser.parse_args()

# 模型参数
gpu_count = args.gpu_count
channels = args.channels
as_gray = args.as_gray
target_size = (256, 256)
# 通道数
color_mode = {
    1: "grayscale",
    3: "rgb"
}

test_root = os.path.join("data", 'mszs_test256_1')

label_path = os.path.join(test_root, 'label')
images = glob.glob(os.path.join(label_path, "*.jpg")) + \
         glob.glob(os.path.join(label_path, "*.png"))


def adjustData(image, mask, plabel, pedge, flag_multi_class, num_class):
    assert mask.size != 0

    image1 = image
    mask1 = mask
    plabel1 = plabel
    pedge1 = pedge

    mask = mask / 255
    plabel = plabel / 255
    pedge = pedge / 255

    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    plabel[plabel > 0.5] = 1
    plabel[plabel <= 0.5] = 0

    pedge[pedge > 0.5] = 1
    pedge[pedge <= 0.5] = 0

    return (image, image1), (mask, mask1), (plabel, plabel1), (pedge, pedge1)


def TotalGenerator(batch_size, train_path, image_folder, mask_folder, plabel_folder, pedge_folder, aug_dict,
                   image_color_mode="grayscale",
                   mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=False, num_class=2, save_to_dir=None, save_format='png', target_size=(256, 256),
                   seed=1):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    plabel_datagen = ImageDataGenerator(**aug_dict)
    pedge_datagen = ImageDataGenerator(**aug_dict)

    plabel_generator = plabel_datagen.flow_from_directory(
        train_path,
        classes=[plabel_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        save_format=save_format,
        seed=seed)

    pedge_generator = pedge_datagen.flow_from_directory(
        train_path,
        classes=[pedge_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    train_generator = zip(image_generator, mask_generator, plabel_generator, pedge_generator)
    for (image, mask, plabel, pedge) in train_generator:
        image, mask, plabel, pedge = adjustData(image, mask, plabel, pedge, flag_multi_class, num_class)
        yield (image, mask, plabel, pedge)


data_gen_args = dict(rotation_range=0.2, width_shift_range=0.05, height_shift_range=0.05,
                     shear_range=0.05, zoom_range=0.05,
                     horizontal_flip=False, fill_mode='nearest')

myGene = TotalGenerator(1, test_root,
                        'image', 'label', 'plabel', 'pedge',
                        data_gen_args,
                        image_color_mode=color_mode.get(channels),
                        save_to_dir=None,
                        target_size=target_size)


def custom_recall(y_true, y_pred):
    """
    召回率
    :param y_true:
    :param y_pred:
    :return:
    """
    assert len(y_true) == len(y_pred)
    y_true = K.constant(y_true)
    y_pred = K.constant(y_pred)

    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # TP
    P = K.sum(K.round(K.clip(y_true, 0, 1)))
    FN = P - TP  # FN=P-TP
    recall = TP / (TP + FN + K.epsilon())  # TP/(TP+FN)
    recall = K.eval(recall)
    return recall


def custom_precision(y_true, y_pred):
    """
    精确率
    :param y_true:
    :param y_pred:
    :return:
    """
    assert len(y_true) == len(y_pred)
    y_true = K.constant(y_true)
    y_pred = K.constant(y_pred)

    # true_positives = K.sum(y_pred * y_true)  # K.round(K.clip(y_pred * y_true, 0, 1)))
    # predicted_positives = K.sum(y_pred)  # K.round(K.clip(y_pred, 0, 1)))
    # precision = true_positives / (predicted_positives + K.epsilon())

    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # TP   #clip:溢出元素改为边界值
    N = (-1) * K.sum(K.round(K.clip(y_true - K.ones_like(y_true), -1, 0)))  # N
    TN = K.sum(K.round(K.clip((y_true - K.ones_like(y_true)) * (y_pred - K.ones_like(y_pred)), 0, 1)))  # TN
    FP = N - TN
    precision = TP / (TP + FP + K.epsilon())  # TP/(TP + FP)      # 用法：返回数值表达式中使用的模糊因子的值。返回：一个浮点数。
    precision = K.eval(precision)
    return precision


precision_list, recall_list, f1_list = list(), list(), list()

for index in range(len(images)):
    image, y_true, pre_y_label, pre_y_edge = next(myGene)
    y_pred = np.maximum(pre_y_label, pre_y_edge, out=None)
    inputss = np.hstack((image[1][0, :, :, :], y_true[1][0, :, :, :], y_pred[1][0, :, :, :]))
    # im = Image.fromarray(np.uint8(pre_y_label[0][0, :, :, :]))
    cv2.imwrite('sample_results_1.png', np.uint8(pre_y_label[0][0, :, :, :] * 255))

    # 计算精确度召回率和f1
    precision = custom_precision(y_true[0], y_pred[0])
    recall = custom_recall(y_true[0], y_pred[0])
    f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    cucl = f"precision:{round(float(precision),2)}, recall:{round(float(recall),2)}, f1:{round(f1,2)}"
    print(cucl)

    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(inputss, cucl, (150, 240), font, 2, (128, 0, 128), 2)

    cv2.imwrite(f'data/xor/{index}.png', inputss)

# 平均值
precision_avg = sum(precision_list) / len(images)
recall_avg = sum(recall_list) / len(images)
f1_avg = sum(f1_list) / len(images)
print(f"precision_avg:{precision_avg},recall_avg:{recall_avg},f1_avg:{f1_avg}")
