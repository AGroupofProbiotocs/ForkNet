# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 11:31:12 2018

@author: Dragon
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.cm as cm
import os
import imageio as imgio
import tensorflow as tf
import math
import cv2

#-----------------------------------------------------------------------------
def image_clip():
    pass

#-----------------------------------------------------------------------------
def view_bar(step, total):
    num = step + 1
    rate = float(num) / float(total)
    rate_num = int(rate * 100)
    arrow = 0 if num==total else 1
    r = '\r[%s%s%s]%d%%' % ('■'*rate_num, '▶'*arrow, '-'*(100-rate_num-arrow), rate * 100)
    sys.stdout.write(r)
    sys.stdout.flush()

#-----------------------------------------------------------------------------
def normalize(data, lower, upper):
    mx = np.max(data)
    mn = np.min(data)
    if mx==mn:
#        print('大小', data.shape)
#        plt.imshow(data[5], cmap='gray')
        norm_data = np.zeros(data.shape)
    else:  
        norm_data = (upper-lower)*(data - mn) / (mx - mn) + lower
    return norm_data

#-----------------------------------------------------------------------------
def Kirsch_filter(img):
    'Kirsch operator for calculating the gradient of image'
    output = np.zeros([img.shape[0], img.shape[1]])
    pad_img = np.zeros([img.shape[0]+2, img.shape[1]+2])
    pad_img[1:-1,1:-1] = img
    
    t_1 = np.reshape([5,5,5,-3,0,-3,-3,-3,-3], (3,3))
    t_2 = np.reshape([-3,5,5,-3,0,5,-3,-3,-3], (3,3))
    t_3 = np.reshape([-3,-3,5,-3,0,5,-3,-3,5], (3,3))
    t_4 = np.reshape([-3,-3,-3,-3,0,5,-3,5,5], (3,3))
    t_5 = np.reshape([-3,-3,-3,-3,0,-3,5,5,5], (3,3))
    t_6 = np.reshape([-3,-3,-3,5,0,-3,5,5,-3], (3,3))
    t_7 = np.reshape([5,-3,-3,5,0,-3,5,-3,-3], (3,3))
    t_8 = np.reshape([5,5,-3,5,0,-3,-3,-3,-3], (3,3))
    templete_group = [t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8]
    
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            pixel_out = []
            for templete in templete_group:
                p = np.sum(pad_img[i:i+3, j:j+3]*templete)
                pixel_out.append(p)
            output[i, j] = max(pixel_out)
    return output

# -----------------------------------------------------------------------------
def aop(x_0, x_45, x_90, x_135, normalization = False):
    '''
    Calculate the AoP
    '''
    AoP = 0.5 * np.arctan((x_45 - x_135) / (x_0 - x_90 + 1e-8))
    if normalization:
        AoP = normalize(AoP,0,1)

    return AoP

#-----------------------------------------------------------------------------
def dolp(x_0, x_45, x_90, x_135, normalization = False):
    '''
    Calculate the DoLP
    '''
    Int = 0.5*(x_0 + x_45 + x_90 + x_135)   
    DoLP = np.sqrt(np.square(x_0-x_90) + np.square(x_45-x_135))/(Int+1e-8)
    DoLP[np.where(Int==0)] = 0   #if Int==0, set the DoLP to 0
    if normalization:
        DoLP = normalize(DoLP,0,1)
    
    return DoLP

#-----------------------------------------------------------------------------
def psnr(ground_truth, ref, mx):
    '''
    Calculate the PSNR
    '''
    diff = ref - ground_truth   
    diff = diff.flatten('C')
    rmse = np.sqrt(np.mean(diff ** 2.))
    PSNR = 20 * np.log10(mx / rmse)

    return PSNR

#------------------------------------------------------------------------------
def pad_shift(bic_img):
    '''
    pad and shift the bicubic images for keeping the position imformation
    '''
    if len(bic_img.shape) == 4:
        pad_width = ((0, 0), (1, 1), (1, 1), (0, 0))
        bic_pad = np.pad(bic_img, pad_width, 'edge')
        bic_shift = np.zeros(bic_img.shape)
        bic_shift[:, :, :, 0] = bic_pad[:, 1:-1, 1:-1, 0]
        bic_shift[:, :, :, 1] = bic_pad[:, 1:-1, 0:-2, 1]
        bic_shift[:, :, :, 2] = bic_pad[:, 0:-2, 1:-1, 2]
        bic_shift[:, :, :, 3] = bic_pad[:, 0:-2, 0:-2, 3]
        return bic_shift
    if len(bic_img.shape) == 3:
        pad_width = ((1, 1), (1, 1), (0, 0))
        bic_pad = np.pad(bic_img, pad_width, 'edge')
        bic_shift = np.zeros(bic_img.shape)
        bic_shift[:, :, 0] = bic_pad[1:-1, 1:-1, 0]
        bic_shift[:, :, 1] = bic_pad[1:-1, 0:-2, 1]
        bic_shift[:, :, 2] = bic_pad[0:-2, 1:-1, 2]
        bic_shift[:, :, 3] = bic_pad[0:-2, 0:-2, 3]
        return bic_shift
    else:
        print('Wrong shape!')


# -----------------------------------------------------------------------------------
def plot_feature_map(conv_img, plot_dir, maps_all=True):
    # get number of convolutional maps
    if maps_all:
        num_maps = conv_img.shape[3]
        maps = range(conv_img.shape[3])
    else:
        maps = np.linspace(0, conv_img.shape[3], 8)
        num_maps = len(maps)

    for map in maps:
        img = normalize(conv_img[0, :, :, map],1,0)
        imgio.imsave(os.path.join(plot_dir, '{}.jpg'.format(map)), img)

#---------------------------------------------------------------------------------------
def fig2array(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGB buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)

    return buf

# -------------------------------------------------------------------------------------
def count_para():
    print('Trainable variables: ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

#----------------------------------------------------------------------------------------
def gaussian_2d_array(size, std=20., mean=None, normalize=True):
    m = np.arange(0, size[0])
    n = np.arange(0, size[1])
    i, j = np.meshgrid(m, n, indexing='ij')
    if mean == None:
        x_mean = size[1] // 2
        y_mean = size[0] // 2
    else:
        x_mean = mean[0]
        y_mean = mean[1]
    z = 1/(2*math.pi*(std**2)) * np.exp(-((j-x_mean)**2 + (i-y_mean)**2)/(2*(std**2)))
    if normalize:
        sum_v = np.sum(z)
        z = z / sum_v
    return z

def gaussian_2d_random_choice(sample_obj, num, std=20., mean=None, replace=False):
    m, n = sample_obj.shape
    prob_map = gaussian_2d_array([m, n], std=std, mean=mean).reshape(-1)
    sample_obj = sample_obj.reshape(-1)
    seleted_idx = np.random.choice(sample_obj, size=num, replace=replace, p=prob_map)
    return seleted_idx

def gs_rand_choice(range_x, range_y, num, idx_map_size=[24, 32], std=16., mean=None, replace=False):
    # only used for current project
    prob_map = gaussian_2d_array(idx_map_size, std=std, mean=mean).reshape(-1)
    patch_idx = np.arange(range_x, range_y)
    seleted_idx = np.random.choice(patch_idx, size=num, replace=replace, p=prob_map)
    return seleted_idx

#------------------------------------------------------------------------------------------
def interpolate_zero(img):
    if len(img.shape) == 2:
        img = img[None, ..., None]
    if len(img.shape) == 3:
        img = img[None, ...]

    batch, h, w, c = img.shape
    output = np.zeros((batch, h, w, 4))

    for i in range(4):
        output[:, i//2::2, i%2::2,  i] = img[:, i//2::2, i%2::2,  0]

    return output

#------------------------------------------------------------------------------------------
def msc2chn(img):

    h, w = img.shape
    output = np.zeros((h//2, w//2, 4))

    for i in range(4):
        output[:, :,  i] = img[i//2::2, i%2::2]

    return output

#------------------------------------------------------------------------------------------
def interpolate_bic(img):

    h, w = img.shape
    output = np.zeros((h, w, 4))
    ds = np.zeros((h//2, w//2, 4))

    for i in range(4):
        mid = img[i//2::2, i%2::2]
        ds[:,:,i] = mid
        output[:, :, i] = cv2.resize(mid, (w, h), cv2.INTER_CUBIC)

    output = pad_shift(output)

    return output, ds

