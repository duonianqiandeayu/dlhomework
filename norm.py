import  numpy
import os
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob


def cal_norm(path):
    img_h, img_w = 64, 64
    means, stdevs = [], []
    img_list = []
    img_fns = glob(os.path.join(path, '*', '*.jpg'))
    print(len(img_fns))
    for sing_img in tqdm(img_fns):
        img = cv2.imread(sing_img)
        # img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)

    imgs = np.concatenate(img_list, axis=1)
    imgs = imgs.astype(np.float32) / 255

    for i in range(3):
        pixels =imgs[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    means.reverse()
    stdevs.reverse()
    print('mean = {}'.format(means))
    print('std = {}'.format(stdevs))

train_data_path = "../ImageData2/train"
val_data_path = "../ImageData2/val"
cal_norm(train_data_path)
cal_norm(val_data_path)