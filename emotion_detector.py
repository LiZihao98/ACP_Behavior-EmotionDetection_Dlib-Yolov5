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
import torch
from models.experimental import attempt_load
# from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
import numpy as np
import cv2

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def predict(frame, model, half=False, device='', imgsz=640, opt_conf_thres=0.25, opt_iou_thres=0.45):
    # Initialize
    device = select_device(device)
    half &= device.type != 'cpu'
    # Load model

    stride = int(model.stride.max())
    names = model.module.names if hasattr(model, 'module') else model.names
    if half:
        model.half()  # to FP16
    # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # init img
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    # warm up
    # if device.type != 'cpu':
    #     model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once

    img = letterbox(frame, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]

    # Inference
    pred = model(img)[0]
    # NMS
    pred = non_max_suppression(pred, opt_conf_thres, opt_iou_thres)

    ret = []
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]}'
                if label != "face":
                    prob = round(float(conf) * 100, 2)  # round 2
                    ret_i = [label, prob, xyxy]
                    ret.append(ret_i)

    return ret
