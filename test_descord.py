# coding=utf-8
# Author  : jiangna
import time

import tensorflow as tf
import argparse

from Models import build_model, color_mode
from metrics.merge_metrics import tf2keras
from utils.data import *
from keras.backend.tensorflow_backend import set_session
import keras.backend as K

EPS = 1e-12

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'  # 使用BFC算法
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)
set_session(session)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="attunet")
parser.add_argument("--exp_name", type=str, default='archive_train256_3')
parser.add_argument("--datatest_name", type=str, default="archive_train256_3")  # attunet-mszs
parser.add_argument("--channels", type=int, default=1)  # 默认单通道
parser.add_argument("--as_gray", type=bool, default=False)  # 3 False , 1 True
parser.add_argument("--input_height", type=int, default=256)
parser.add_argument("--input_width", type=int, default=256)

parser.add_argument("--gpu_count", type=int, default=2)
parser.add_argument("--n_classes", type=int, default=2)
args = parser.parse_args()

# 模型参数
model_name = args.model_name
n_class = args.n_classes
input_height = args.input_height
input_width = args.input_width
gpu_count = args.gpu_count
channels = args.channels
as_gray = args.as_gray
target_size = (input_height, input_width)
resize_op = 1
# test_root = os.path.join("data", 'mstest')

train_save_path = os.path.join('weights', args.exp_name, model_name)
model_names = os.path.join(train_save_path, 'unet_membrane.hdf5')

# model
# model = build_model(model_name, n_class, channels,
#                     input_height=input_height,
#                     input_width=input_width)
# model.load_weights(model_names)


test_root = os.path.join("data", 'output', 'attunet', 'deeptest')
images_path = os.path.join(test_root, 'image')
segs_path = os.path.join(test_root, 'label')

images = glob.glob(os.path.join(images_path, "*.jpg")) + \
         glob.glob(os.path.join(images_path, "*.png"))

data_gen_args = dict(rotation_range=0.2, width_shift_range=0.05, height_shift_range=0.05,
                     shear_range=0.05, zoom_range=0.05,
                     horizontal_flip=True, fill_mode='nearest')

myGene = trainGenerator(2, test_root,
                        'image', 'label',
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


precision_list, recall_list, f1_list \
    = list(), list(), list()

for _ in range(len(images)):
    y_pred, y_true = next(myGene)
    # y_pred = model.predict(img_source)

    print(y_pred.shape)
    print(y_true.shape)

    # 计算精确度召回率和f1
    precision = custom_precision(y_true, y_pred)
    recall = custom_recall(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    print(f"precision:{precision},recall:{recall},f1:{f1}")

# 平均值
precision_avg = sum(precision_list) / len(images)
recall_avg = sum(recall_list) / len(images)
f1_avg = sum(f1_list) / len(images)
print(f"precision_avg:{precision_avg},recall_avg:{recall_avg},f1_avg:{f1_avg}")
