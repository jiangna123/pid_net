#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 9/16/21 9:41 AM
# File    : custom_metrics.py
# Describe: 自定义metrics
# Author  : jiangna



import tensorflow as tf
from keras.utils import get_custom_objects

__all__ = ['iou_score', 'precision', 'recall', 'jaccard_score', 'f1_score', 'f2_score',
           'dice_score', 'get_f_score', 'get_iou_score', 'get_jaccard_score']

SMOOTH = 1.

import keras.backend as K


def precision_threshold(threshold=0.5):
    def precision(y_true, y_pred):
        """
        使用阈值计算精度
         #  clip溢出的元素改为边界值
        # greater 比较第一个参数个第二个参数的大小，返回布尔列表
        # cast 把数据转为float32
        """
        threshold_value = threshold
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))  # TP
        predicted_positives = K.sum(y_pred)
        precision_ratio = true_positives / (predicted_positives + K.epsilon())

        return precision_ratio

    return precision


# model.compile(..., metrics = [precision_threshold(0.1), precision_threshold(0.2),precision_threshold(0.8), recall_threshold(0.2,...)])

def precision(y_true, y_pred):
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


def false_alarm(y_true, y_pred):
    """
    虚警率（False alarm）表示负类样本被分为正类样本在所有负类样本中的比例
    :param y_true:
    :param y_pred:
    :return:
    """
    N = (-1) * K.sum(K.round(K.clip(y_true - K.ones_like(y_true), -1, 0)))  # N
    TN = K.sum(K.round(K.clip((y_true - K.ones_like(y_true)) * (y_pred - K.ones_like(y_pred)), 0, 1)))  # TN
    FP = N - TN
    falarm = FP / (TN + FP + K.epsilon())  # FP/(TN + FP)
    return falarm


def mis_alarm(y_true, y_pred):
    """
    漏警率表示（Missing alarm）表示正类样本被分为负类样本在所有正类样本中的比例
    :param y_true:
    :param y_pred:
    :return:
    """
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # TP
    P = K.sum(K.round(K.clip(y_true, 0, 1)))
    FN = P - TP  # FN = P-TP
    misarm = FN / (TP + FN + K.epsilon())  # FN/(TP+FN)
    return misarm


def recall(y_true, y_pred):
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


# ============================ Jaccard/IoU score ============================
def iou_score(gt, pr, class_weights=1., smooth=SMOOTH, per_image=True, threshold=None):
    '''
    参数：
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison),
        if ``None`` prediction prediction will not be round
    返回：
        IoU/Jaccard score in range [0, 1]
    '''
    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]

    if threshold is not None:
        pr = tf.greater(pr, threshold)
        pr = tf.cast(pr, dtype=tf.float32)

    intersection = tf.reduce_sum(gt * pr, axis=axes)
    union = tf.reduce_sum(gt + pr, axis=axes) - intersection
    iou = (intersection + smooth) / (union + smooth)

    # mean per image
    if per_image:
        iou = tf.reduce_mean(iou, axis=0)

    # weighted mean per class
    iou = tf.reduce_mean(iou * class_weights)

    return iou


# 计算IOU得分
def get_iou_score(class_weights=1., smooth=SMOOTH, per_image=True, threshold=None):
    def score(gt, pr):
        return iou_score(gt, pr, class_weights=class_weights, smooth=smooth, per_image=per_image, threshold=threshold)

    return score


jaccard_score = iou_score
get_jaccard_score = get_iou_score

get_custom_objects().update({'iou_score': iou_score,
                             'jaccard_score': jaccard_score})


# ============================== F/Dice - score ==============================

def f_score(gt, pr, class_weights=1, beta=1, smooth=SMOOTH, per_image=True, threshold=None):
    # F_score（Dice系数）可以解释为精确度和召回率的加权平均值，
    # 其中F-score在1时达到其最佳值，在0时达到最差分数。
    # 精确率和召回率对F1-score的相对影响是一样的，公式表示为：
    # $F_\beta(precision, recall) = (1 + \beta^2) \frac{precision \cdot recall}
    #    {\beta^2 \cdot precision + recall}$
    # 公式还有另外一种表达形式：
    # $F_\beta(A, B) = \frac{(1 + \beta^2) TP} {(1 + \beta^2) TP + \beta^2 FN + FP}$
    # 其中 TP表示ture positive
    # FP表示fasle positive
    # FN表示false negtive
    # 参数：
    #   gt: ground truth 4D keras tensor (B, H, W, C)
    #    pr: prediction 4D keras tensor (B, H, W, C)
    #    class_weights: 1. or list of class weights, len(weights) = C
    #    beta: f-score coefficient
    #    smooth: value to avoid division by zero
    #    per_image: if ``True``, metric is calculated as mean over images in batch (B),
    #        else over whole batch
    #    threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction prediction will not be round
    # 返回：
    # [0, 1]区间内的F-score

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

    # weighted mean per class
    score = tf.reduce_mean(score * class_weights)

    return score


def get_f_score(class_weights=1, beta=1, smooth=SMOOTH, per_image=True, threshold=None):
    '''
    参数:
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        beta: f-score coefficient
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction prediction will not be round
    返回:
        ``callable``: F-score
    '''

    def score(gt, pr):
        return f_score(gt, pr, class_weights=class_weights, beta=beta, smooth=smooth, per_image=per_image,
                       threshold=threshold)

    return score


f2_score = get_f_score(beta=2)
f1_score = get_f_score(beta=1)
dice_score = f1_score

# Update custom objects
get_custom_objects().update({
    # 'f1_score': f1_score,
    # 'f2_score': f2_score,
    'dice_score': dice_score,
    'precision': precision,
    'recall': recall,
    'mis_alarm': mis_alarm,
    'false_alarm': false_alarm
})








 