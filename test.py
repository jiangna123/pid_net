#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 1/19/21 11:25 AM
# File    : test.py
# Describe:预测图片(多GPU版),不带edge
# Author  : jiangna

import tensorflow as tf
import argparse
from keras.utils import multi_gpu_model
from Models import build_model
from utils.data import *
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
parser.add_argument("--datatest_name", type=str, default="archive_test256_3")
parser.add_argument("--channels", type=int, default=3)  # 默认单通道
parser.add_argument("--as_gray", type=bool, default=False)  # 3 False , 1 True
parser.add_argument("--input_height", type=int, default=256)
parser.add_argument("--input_width", type=int, default=256)

parser.add_argument("--gpu_count", type=int, default=2)
parser.add_argument("--n_classes", type=int, default=2)
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

test_root = os.path.join("data", args.datatest_name)
train_save_path = os.path.join('weights', 'exp1', model_name)
model_names = os.path.join(train_save_path, 'unet_membrane.hdf5')

images = glob.glob(os.path.join(test_root, "*.jpg")) + glob.glob(
    os.path.join(test_root, "*.png")) + glob.glob(os.path.join(test_root, "*.jpeg"))
images.sort()

img_len = len(images)

model = build_model(model_name,
                    n_classes,
                    channels,
                    input_height=input_height,
                    input_width=input_width)
try:
    model = multi_gpu_model(model, gpus=gpu_count)
except Exception as e:
    model = model

model.load_weights(model_names)
testGene = testGenerator(images, target_size=target_size, as_gray=as_gray)
results = model.predict_generator(testGene, img_len, verbose=1)
saveResult("data/output", results)
