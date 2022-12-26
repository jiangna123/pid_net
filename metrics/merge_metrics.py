#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 9/17/21 9:29 AM
# File    : merge_metrics.py
# Author  : jiangna
import tensorflow as tf
from keras.utils import get_custom_objects
import sklearn.metrics as skl
import keras.backend as K
import functools

__all__ = ['precision', 'recall', 'f1_score', 'auc', 'sensitivity', 'specificity',
           'roc_auc', 'mean_iou', 'custom_precision', 'custom_recall', 'cf1_score']


# ===============================自带metrics==================
def tf2keras(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        """tf转keras"""

        value, update_op = method(self, *args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value

    return wrapper


@tf2keras
def sensitivity(y_true, y_pred):
    return tf.metrics.specificity_at_sensitivity(y_true, y_pred, sensitivity=0.5)


@tf2keras
def specificity(y_true, y_pred):
    return tf.metrics.sensitivity_at_specificity(y_true, y_pred, specificity=0.5)


@tf2keras
def mean_iou(y_true, y_pred):
    return tf.metrics.mean_iou(y_true, y_pred, 2)


@tf2keras
def recall(y_true, y_pred):
    return tf.metrics.recall(y_true, y_pred)


@tf2keras
def precision(y_true, y_pred):
    return tf.metrics.precision(y_true, y_pred)


@tf2keras
def auc(y_true, y_pred):
    return tf.metrics.auc(y_true, y_pred)


def roc_auc(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=1)
    return tf.py_func(skl.roc_auc_score, (y_true, y_pred), tf.double)


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))  # 2precision * recall/ precision + recall


# ================================自定义metrics================

def f_score(gt, pr, class_weights=1, beta=1, smooth=1., per_image=True, threshold=None):
    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]

    if threshold is not None:
        pr = tf.greater(pr, threshold)
        pr = tf.cast(pr, dtype=tf.float32)

    tp = tf.reduce_sum(gt * pr, axis=axes)
    fp = tf.reduce_sum(pr, axis=axes) - tp
    fn = tf.reduce_sum(gt, axis=axes) - tp

    score = ((1 + beta ** 2) * tp + smooth) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)

    # mean per image
    if per_image:
        score = tf.reduce_mean(score, axis=0)
    score = tf.reduce_mean(score * class_weights)  # weighted mean per class

    return score


def custom_f_score(class_weights=1, beta=1, smooth=1., per_image=True, threshold=None):
    def score(gt, pr):
        return f_score(gt, pr, class_weights=class_weights, beta=beta, smooth=smooth, per_image=per_image,
                       threshold=threshold)

    return score


def custom_precision(y_true, y_pred):
    """
    精确率
    :param y_true:
    :param y_pred:
    :return:
    """
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # TP   #clip:溢出元素改为边界值
    N = (-1) * K.sum(K.round(K.clip(y_true - K.ones_like(y_true), -1, 0)))  # N
    TN = K.sum(K.round(K.clip((y_true - K.ones_like(y_true)) * (y_pred - K.ones_like(y_pred)), 0, 1)))  # TN
    FP = N - TN
    precision = TP / (TP + FP + K.epsilon())  # TP/(TP + FP)      # 用法：返回数值表达式中使用的模糊因子的值。返回：一个浮点数。
    return precision


def custom_recall(y_true, y_pred):
    """
    召回率
    :param y_true:
    :param y_pred:
    :return:
    """
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # TP
    P = K.sum(K.round(K.clip(y_true, 0, 1)))
    FN = P - TP  # FN=P-TP
    recall = TP / (TP + FN + K.epsilon())  # TP/(TP+FN)
    return recall


custom_f1_score = custom_f_score(beta=1)
cf1_score = custom_f1_score
# =============================================


get_custom_objects().update({
    'precision': precision,
    'recall': recall,
    'f1_score': f1_score,
    'auc': auc,
    'sensitivity': sensitivity,
    'specificity': specificity,
    'roc_auc': roc_auc,
    'mean_iou': mean_iou,
    'custom_precision': custom_precision,
    'custom_recall': custom_recall,
    'cf1_score': cf1_score,

})
