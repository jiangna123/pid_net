#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 9/17/21 9:57 AM
# File    : matpl_single_test.py
# Describe: 第一版可视化测试集结果,不带图片显示
# Author  : jiangna
import tensorflow as tf
import argparse

from keras.backend import clear_session
from keras.utils import multi_gpu_model
import matplotlib.pyplot as plt

from Models import build_model, color_mode
from metrics.merge_metrics import custom_recall
from utils.data import *
from utils.utils import mk_if_not_exits
from keras.backend.tensorflow_backend import set_session

# 设置gpu运行参数

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'  # 使用BFC算法
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)
set_session(session)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="attunet")
parser.add_argument("--exp_name", type=str, default='mszs_train256_1')
parser.add_argument("--datatest_name", type=str, default="mszs_train256_1")  # attunet-mszs
parser.add_argument("--channels", type=int, default=1)  # 默认单通道
parser.add_argument("--as_gray", type=bool, default=True)  # 3 False , 1 True
parser.add_argument("--input_height", type=int, default=256)
parser.add_argument("--input_width", type=int, default=256)

parser.add_argument("--gpu_count", type=int, default=2)
parser.add_argument("--n_classes", type=int, default=1)
args = parser.parse_args()

# 模型参数
model_name = args.model_name
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
gpu_count = args.gpu_count
channels = args.channels
as_gray = args.as_gray
target_size = (input_height, input_width)

test_root = os.path.join("data", 'test')
train_save_path = os.path.join('weights', args.exp_name, model_name)
model_names = os.path.join(train_save_path, 'unet_membrane.hdf5')

images = glob.glob(os.path.join(test_root, "*.jpg")) + glob.glob(
    os.path.join(test_root, "*.png")) + glob.glob(os.path.join(test_root, "*.jpeg"))
images.sort()

img_len = len(images)

clear_session()
model = build_model(model_name, n_classes, channels,
                    input_height=input_height,
                    input_width=input_width)
model.load_weights(model_names)
data_gen_args = dict(rotation_range=0.2, width_shift_range=0.05, height_shift_range=0.05,
                     shear_range=0.05, zoom_range=0.05,
                     horizontal_flip=True, fill_mode='nearest')

myGene = trainGenerator(2, test_root,
                        'image', 'label',
                        data_gen_args,
                        image_color_mode=color_mode.get(channels),
                        save_to_dir=None,
                        target_size=target_size)

for img, y_true in myGene:
    y_pred = model.predict(img)

    # print(y_pred.shape)
    # print(y_true.shape)
    # y_pred = y_pred.argmax(axis=-1)
    # y_true = y_true.argmax(axis=-1)
    # print(y_pred.shape)
    # print(y_true.shape)
    # y_pred = y_pred.flatten()
    # y_true = y_true.flatten()
    # # recall = tf.metrics.recall(y_true, y_pred)
    # recall = tf.metrics.recall(y_true, y_pred)
    # with tf.Session() as sess:
    #     print(sess.run(recall))
    # # print(recall)
    # # print(recall)

    pred_p = (y_pred > 0).sum()
    true_p = (y_true * y_pred > 0).sum()
    precision = true_p / pred_p
    print("Precision :%1.4f" % (precision))
