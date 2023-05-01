# import the necessary packages
import numpy as np
import cv2
import glob
import os
from tqdm import tqdm
import math
from PIL import Image
import pandas as pd


def crop_image(image):
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,gray = cv2.threshold(gray,25,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print('no contours!')
        flag = 0
        return image, flag
    cnt = max(contours, key=cv2.contourArea)
    ((x, y), r) = cv2.minEnclosingCircle(cnt)
    x = int(x)
    y = int(y)
    r = int(r)
    flag = 1
    print(y,x,r)
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

# 6323_left
# 15066_left
img = cv2.imread(f'./dataset/train/8421_left.jpeg')
if img == None:
    print("fuck")
img_cropped = crop_image(img)


# img_cropped = cv2.resize(img_cropped, (512, 512), interpolation=cv2.INTER_NEAREST)
# print(img.size)
# print(img_cropped.size)
cv2.namedWindow('My Image', cv2.WINDOW_NORMAL)
cv2.imshow('My Image', img_cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
