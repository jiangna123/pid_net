#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 9/16/21 9:41 AM
# File    : custom_metrics.py
# Describe: metrics
# Author  : jiangna
import tensorflow as tf
from keras.callbacks import Callback
from keras.utils import get_custom_objects
import sklearn.metrics as skl
import keras.backend as K
import functools

__all__ = ['precision', 'recall', 'f1_score', 'auc', 'sensitivity', 'specificity',
           'roc_auc', 'mean_iou']


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.x_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, log={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.x_val, verbose=0)
            score = skl.roc_auc_score(self.y_val, y_pred)
            print('\n ROC_AUC - epoch:%d - score:%.6f \n' % (epoch + 1, score))


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


get_custom_objects().update({
    'precision': precision,
    'recall': recall,
    'f1_score': f1_score,
    'auc': auc,
    'sensitivity': sensitivity,
    'specificity': specificity,
    'roc_auc': roc_auc,
    'mean_iou': mean_iou
})
