# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 21:07:21 2018

@author: Xianglong Zeng
"""
import numpy as np

img_num = 110

#split the images into training set and validation set 
# img_index = list(range(img_num))
# np.random.shuffle(img_index)
# img_index = str(img_index)[1:-1]
img_index = list(np.random.permutation(img_num))
train_img_index = str(img_index[:100])[1:-1]
val_img_index = str(img_index[100:])[1:-1]

with open('../list/train_image_index.list', 'w') as f:
    f.write(train_img_index)

with open('../list/val_image_index.list', 'w') as f:
    f.write(val_img_index)
