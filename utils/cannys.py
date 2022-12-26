#!/usr/bin/env python
# -*- coding: utf-8 -*-
# file:cannys
# Author  : jiangna
# desctription： 处理边缘化标签
# datetime： 22-10-28 下午1:17 
# ============================

import os

import cv2
import numpy as np

# 1.高斯滤波去噪声
# 2.sobel计算梯度及角度
# g =np.sqrt(Gx**2 + Gy**2)
# theta = np.tan(Gy/Gx)
# eg: P5sobel=|P5x|+|P5y|=|(p3-p2)+2(p6-p5)+(p9-p7)|+|(p7-p1)+2(p8-p2)+(p9-p3)|
# 3.极大值抑制
# 4.阈值滞后


windowname = "OpenCV Media Player"
cv2.namedWindow(windowname)


def read_image(img, png, dest):
    filename, filespl = os.path.splitext(img)

    image = cv2.imread(img)

    # 边缘检测
    thresh = cv2.Canny(image, 32, 256)
    thresh, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    """
    thresh: 目标图像
    contours: 轮廓本身
    hierarchy:
    """
    save_path = f"{dest}/{png}"
    cv2.imwrite(save_path, thresh)

    # # 轮廓绘制
    # save_path = f"{dest}/{filename}_mark{filespl}"
    # img = cv2.drawContours(image, contours, -1, (0, 0, 255,), 3)
    # cv2.imwrite(save_path, img)


# read_image('train_data_21_0.png', ".")
# cv2.waitKey()
#
# cv2.destroyWindow(windowname)


base_path = os.path.dirname(os.path.dirname(__file__))
# label_path = f"{base_path}/data/mszs_train256_1/val_label"
# edge_path = f"{base_path}/data/mszs_train256_1/val_edge"

label_path = f"{base_path}/data/archive_train256_3/val_label"
edge_path = f"{base_path}/data/archive_train256_3/val_edge"

for png in os.listdir(label_path):
    read_image(f"{label_path}/{png}", png, edge_path)
