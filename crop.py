# import the necessary packages
import numpy as np
import cv2
import glob
import os
from tqdm import tqdm
import math
from PIL import Image
import pandas as pd

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)

def crop_image(image):
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,gray = cv2.threshold(gray,25,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print('no contours!')
        return cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)
    cnt = max(contours, key=cv2.contourArea)
    ((x, y), r) = cv2.minEnclosingCircle(cnt)
    x = int(x)
    y = int(y)
    r = int(r)
    H, W, C = output.shape
    if r > 100:
        if y - r < 0 or y + r > H:
            y_min = 0
            y_max = H
        else:
            y_min = y - r
            y_max = y + r

        if x - r < 0 or x + r > W:
            x_min = 0
            x_max = W
        else:
            x_min = x - r
            x_max = x + r

        output = output[y_min:y_max, x_min:x_max]
        if output.size == 0:
            output = cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)
        else:
            output = cv2.resize(output, (512, 512), interpolation=cv2.INTER_NEAREST)
        return output
    else:
        output = cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)
        return output


img_name, label = getData('train')

for index in range(0, len(img_name)):
    print(index, img_name[index])
    img = cv2.imread(f'./dataset/train/{img_name[index]}.jpeg')
    if img is None:
        continue
    img_cropped = crop_image(img)
    cv2.imwrite(f'./new_dataset/train/{img_name[index]}.jpeg', img_cropped)
