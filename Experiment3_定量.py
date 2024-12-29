#!/usr/bin/env python
import cv2

import numpy as np
import scipy.misc
from os import listdir
from os.path import join
import math
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

out_path = 'output/B'
orig_path = 'D:/PyCharm/PyProjects/Cycle-SNSPGAN-tttest/datasets/hazy2clear/test/reB'

def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat'])


def load_all_image(path):
    return [join(path, x) for x in listdir(path) if is_image_file(x)]

datas = load_all_image(out_path)
gts = load_all_image(orig_path)

datas.sort()
gts.sort()

ssims = []
psnrs = []
#mses = []
for i in range(len(datas)):#
    out= cv2.imread(datas[i])
#    out = cv2.resize(out, (308, 512))
    gt = cv2.imread(gts[i])
#    gt = cv2.resize(gt, (308, 512))
    grayA = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    ssims.append(compare_ssim(grayA, grayB,win_size=7))
    psnrs.append(compare_psnr(out, gt))

print(np.round(np.mean(ssims),3),'&',np.round(np.mean(psnrs),2))

