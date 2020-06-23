# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 16:33:21 2018

@author: Xianglong Zeng
"""
import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image
from .utils import Kirsch_filter, dolp, aop, normalize, view_bar, pad_shift
import h5py
import cv2
import math

IMG_NUM = 110
IMG_WIDTH = 1280
IMG_HEIGHT = 960
PATCH_SIZE = 40
CLIP_STRIDES = 40
#PATCH_NUM = 10
BIC_filename = '../data/training_set/BIC.h5'
Y_filename = '../data/training_set/Y.h5'
labels_filename = '../data/training_set/Labels.h5'
data_path = '/home/data_set/Polarized_Images/training_set/'

# the clip coordinates on x axis
x_steps = np.arange(0, IMG_WIDTH-PATCH_SIZE, CLIP_STRIDES)
if (IMG_WIDTH-PATCH_SIZE)/CLIP_STRIDES != 0: 
    x_steps = np.append(x_steps, IMG_WIDTH-PATCH_SIZE)
x_num = len(x_steps)
# the clip coordinates on y axis   
y_steps = np.arange(0, IMG_HEIGHT-PATCH_SIZE, CLIP_STRIDES) 
if (IMG_HEIGHT-PATCH_SIZE)/CLIP_STRIDES != 0:
    y_steps = np.append(y_steps, IMG_HEIGHT-PATCH_SIZE)
y_num = len(y_steps)

PATCH_NUM = IMG_NUM * x_num * y_num
Para = np.zeros([PATCH_NUM, PATCH_SIZE, PATCH_SIZE, 3], np.float32)
X = np.zeros([PATCH_NUM, PATCH_SIZE, PATCH_SIZE, 4], np.float32)
Y = np.zeros([PATCH_NUM, PATCH_SIZE, PATCH_SIZE, 1], np.float32)
BIC = np.zeros([PATCH_NUM, PATCH_SIZE, PATCH_SIZE, 4], np.float32)

# define the index for downsampling
m = np.arange(0, PATCH_SIZE-1, 2)
n = np.arange(1, PATCH_SIZE, 2)
i_0, j_0 = np.meshgrid(m, m, indexing='ij')
i_45, j_45 = np.meshgrid(m, n, indexing='ij')
i_90, j_90 = np.meshgrid(n, m, indexing='ij')
i_135, j_135 = np.meshgrid(n ,n, indexing='ij')
ds_index = [(i_0, j_0),(i_45, j_45),(i_90, j_90),(i_135, j_135)]

for l in range(1, IMG_NUM+1): 
    for i in range(0, 4): 
        patch_count = 0 
        path_origin = data_path + '/image_{}_{}.bmp'.format(l,i*45)
        img_origin = np.array(Image.open(path_origin), np.float32)/255.
        for j in x_steps:
            for k in y_steps:
                patch = img_origin[k:k+PATCH_SIZE, j:j+PATCH_SIZE]
                patch_ds = patch[ds_index[i]]                                          
                
                batch_index = (l-1) * x_num * y_num + patch_count
                Y[batch_index,ds_index[i][0],ds_index[i][1],0] = patch_ds
                BIC[batch_index,:,:,i] = cv2.resize(patch_ds, patch.shape, cv2.INTER_CUBIC)
                X[batch_index,:,:,i] = patch
                
                patch_count += 1

    view_bar(l, IMG_NUM + 1)
   
Para[:,:,:,0] = 1/2*(X[:,:,:,0] + X[:,:,:,1] + X[:,:,:,2] + X[:,:,:,3])
Para[:,:,:,1] = dolp(X[:,:,:,0], X[:,:,:,1], X[:,:,:,2], X[:,:,:,3])
Para[:,:,:,2] = aop(X[:,:,:,0], X[:,:,:,1], X[:,:,:,2], X[:,:,:,3]) + math.pi/4.
print('min_aop: ', np.min(Para[:,:,:,2]))
print('max_aop: ', np.max(Para[:,:,:,2]))

#Y = Y/255.
BIC = pad_shift(BIC)

#label = np.array([X, Para], dtype=np.float32)

with h5py.File(labels_filename, 'w') as h1:
    h1.create_dataset('labels', data=Para)
with h5py.File(Y_filename, 'w') as h2:
    h2.create_dataset('inputs', data=Y)
with h5py.File(BIC_filename, 'w') as h3:
    h3.create_dataset('bic', data=BIC)


plt.figure()
plt.subplot(2,2,1)
plt.imshow(BIC[350,:,:,0], cmap='gray', label='BIC')
plt.subplot(2,2,2)
plt.imshow(Para[350,:,:,1], cmap='gray', label='DoLP')
plt.subplot(2,2,3)
plt.imshow(Y[350,:,:,0], cmap='gray', label='Input')
plt.subplot(2,2,4)
plt.imshow(X[350,:,:,0], cmap='gray', label='Origin')
plt.savefig('../test.pdf')
print('finished!')

