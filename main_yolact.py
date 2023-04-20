import cv2
import numpy as np
import argparse
from yolact_onnx import COCO_CLASSES, colors, Yolact

if __name__=='__main__':
    myyolact = Yolact(model_path="convert-onnx/models/yolact_resnet50_54_800000.onnx", confThreshold=0.5, nmsThreshold=0.5, keep_top_k=200)
    srcimg = cv2.imread("000000046804.jpg")
    srcimg = myyolact.detect(srcimg)

    cv2.namedWindow('yolact-detect', cv2.WINDOW_NORMAL)
    cv2.imshow('yolact-detect', srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()