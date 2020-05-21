# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 21:07:21 2018

@author: Dragon
"""
import numpy as np

img_num = 300

#随机打乱图片序号并划分训练及验证集
img_index = list(range(img_num)) 
np.random.shuffle(img_index)
img_index = str(img_index)[1:-1]

with open('../list/image_index.list', 'w') as f:
    f.write(img_index)