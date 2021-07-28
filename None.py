from __future__ import print_function
import os,csv
import cv2
import resnet
import torch
import numpy as np
import config


def get_test_list(pair_list):
    with open(pair_list, 'r',encoding="utf-8") as fd:
        pairs = fd.readlines()
    data_list = []
    label_list = []
    mask_list = []
    for pair in pairs:
        splits = pair.split()
        data_list.append(splits[0])
        mask_list.append(splits[1])
        label_list.append(splits[2])

    return data_list,mask_list,label_list

def load_image(img_pathXR,mask_path):
    for i, img_path in enumerate(img_pathXR):
        image = cv2.imread(img_path, 0)
        if image is None:
            print(img_path)
    

if __name__ == '__main__':
    opt = config.Config()
    imgpath_XR,mask_XR,label_XR = get_test_list(opt.mask_FilenameXR)
    imgpath_XT,mask_XT,label_XT = get_test_list(opt.mask_FilenameXT)
    load_image(imgpath_XR, mask_XR)
