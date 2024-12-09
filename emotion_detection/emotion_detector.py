#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
File: emotion_detector.py

Description: This module contains the EmotionDetector class, which utilizes the YOLOv5 model
to detect emotions from images. The class handles loading the model, processing images, and
interpreting the results to categorize emotions such as happy, sad, angry, or surprised.

Classes:
    EmotionDetector: Detects emotions in images using YOLOv5.
"""
import cv2
import numpy as np

def letterbox(img_src, dst_size=(640, 640), pad_color=(114, 114, 114),auto = True):
    #Resize the image while maintaining the aspect ratio.
    #param img_src:   The source image (NumPy array)
    #param dst_size:    The target size (height, width)
    #param pad_color:   The fill color for padding, default is gray
    #return:            The resized image with maintained aspect ratio and padding

    src_h, src_w = img_src.shape[:2]
    dst_h, dst_w = dst_size

    # Scale ratio (new / old)
    r = min(dst_h / src_h, src_w / dst_w)

    # scaleup = True
    #if not scaleup:  # only scale down, do not scale up (for better test mAP)
        #r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(src_w * r)), int(round(src_h* r))
    dw, dh = dst_w - new_unpad[0],dst_h - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    #把原来图片的h和w反过来
    if (src_w, src_h) != new_unpad:  # resize
        img_src = cv2.resize(img_src, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_src = cv2.copyMakeBorder(img_src, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)  # add border
    return img_src, ratio, (dw, dh)


