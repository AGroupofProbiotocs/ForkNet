"""
========================================================================
ForkNet for DoFP sensor to reconstruct s0, dolp and aop, Version 1.0
Copyright(c) 2020  Xianglong Zeng, Yuan Luo, Wenbin Ye
All Rights Reserved.
----------------------------------------------------------------------
Permission to use, copy, or modify this software and its documentation
for educational and research purposes only and without fee is here
granted, provided that this copyright notice and the original authors'
names appear on all copies and supporting documentation. This program
shall not be used, rewritten, or adapted as the basis of a commercial
software or hardware product without first obtaining permission of the
authors. The authors make no representations about the suitability of
this software for any purpose. It is provided "as is" without express
or implied warranty.
----------------------------------------------------------------------
Please cite the following paper when you use it:

Xianglong Zeng, Yuan Luo, Xiaojing Zhao, and Wenbin Ye, "An end-to-end 
fully-convolutional neural network for division of focal plane sensors 
to reconstruct S0, DoLP, and AoP," Opt. Express 27, 8566-8577 (2019)
========================================================================
"""
import tensorflow as tf
import numpy as np
from utils.layers import conv2d, conv2d_bn
import math

def ForkNet(inputs, padding = 'VALID', name='ForkNet'):
    '''
    Built the ForkNet model.
    Args:
        inputs: mosaic polarized images
        padding: padding mode of convolution
    Returns:
        s0: reconstructed s0 images
        dolp: reconstructed dolp images
        aop: reconstructed aop images
        
    '''
#    keep_prob = tf.where(is_training, 0.2, 1.0)
    with tf.variable_scope(name):
        # conventional layers
        x_1 = conv2d(inputs, [4, 4], 96, activation=tf.nn.relu, padding=padding, name='conv_1')
        tf.add_to_collection('feature_maps', x_1)
        x_2 = conv2d(x_1, [3,3], 48, activation=tf.nn.relu, padding=padding, name='conv_2')
        tf.add_to_collection('feature_maps', x_2)

        x_3_1 = conv2d(x_2, [3, 3], 32, activation=tf.nn.relu, padding=padding, name='conv_3_1')
        tf.add_to_collection('feature_maps', x_3_1)
        s0 = conv2d(x_3_1, [5, 5], 1, activation=None, padding=padding, name='conv_4_1')

        x_3_2 = conv2d(x_2, [3, 3], 32, activation=tf.nn.relu, padding=padding, name='conv_3_2')
        tf.add_to_collection('feature_maps', x_3_2)
        dolp = conv2d(x_3_2, [5, 5], 1, activation=None, padding=padding, name='conv_4_2')

        x_3_3 = conv2d(x_2, [3, 3], 32, activation=tf.nn.relu, padding=padding, name='conv_3_3')
        tf.add_to_collection('feature_maps', x_3_3)
        aop = conv2d(x_3_3, [4, 4], 1, activation=None, padding=padding, name='conv_4_3')
        aop = tf.atan(aop) / 2. + math.pi / 4

    return s0, dolp, aop


def MAE_LOSS(s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true):
    '''
    Define the mae loss function.
    '''
    loss = tf.reduce_mean(0.1*tf.abs(s0_true - s0_pred) + tf.abs(dolp_true - dolp_pred) + 0.05*tf.abs(aop_true - aop_pred))

    return loss

def smooth_loss(s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true):
    '''
    Define the smooth loss function.
    '''
    loss = 0.1*smooth_l1_loss(s0_true, s0_pred, 2) + smooth_l1_loss(dolp_true, dolp_pred, 2) + 0.01*smooth_l1_loss(aop_true, aop_pred, 2)

    return loss

def smooth_l1_loss(reg_true, reg_pred, sigma):
    '''

    :param reg_true: a tensor with shape (#object, 4)
    :param reg_pred: a tensor with shape (#object, 4)
    :param sigma:
    :return: smooth_l1 loss
    '''
    sigma = sigma**2
    thres = 1. / sigma
    diff = reg_true - reg_pred

    l1 = tf.abs(diff) - 0.5 / sigma
    l2 = sigma / 2. * tf.square(diff)
    loss = tf.reduce_mean(tf.where(tf.less(tf.abs(diff), thres), l2, l1))

    return loss

def LOSS(s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true, max_value=math.pi/2.):
    '''
    Define the loss function.
    '''
    L, C, S, sl = ssim_loss(aop_true, aop_pred, mv=max_value)
    loss = tf.reduce_mean(tf.reduce_sum(0.1*tf.abs(s0_true - s0_pred) + tf.abs(dolp_true - dolp_pred) + 0.05*tf.abs(aop_true - aop_pred), axis=[1,2,3])) \
                          - 0.02*tf.log(C)
    return loss

def MSE_LOSS(s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true, max_value=math.pi/2.):
    '''
    Define the mse loss function.
    '''
    L, C, S, sl = ssim_loss(aop_true, aop_pred, mv=max_value)
    loss = tf.reduce_mean(0.1*tf.square(s0_true - s0_pred) + tf.square(dolp_true - dolp_pred) + 0.032*tf.square(aop_true - aop_pred)) - 0.02*tf.log(C)
    return loss

def std_variance(x):
    '''
    Compute the std_variance.
    '''
    shape = tf.cast(tf.shape(x), dtype = tf.float32)
    var = tf.sqrt(tf.reduce_sum((x-tf.reduce_mean(x))**2) / (shape[0]*shape[1]*shape[2]*shape[3]-1))

    return var

def covariance(x, y):
    '''
    Compute the covariance.
    '''
    shape = tf.cast(tf.shape(x), dtype=tf.float32)
    covar = tf.reduce_sum((x-tf.reduce_mean(x))*(y-tf.reduce_mean(y))) / (shape[0]*shape[1]*shape[2]*shape[3]-1)

    return covar

def ssim_loss(x, y, mv, k1=0.01, k2=0.03):
    '''
    Define the SSIM loss.
    '''
    c1 = (k1*mv)**2.
    c2 = (k2*mv)**2.
    c3 = c2/2.

    x_mean = tf.reduce_mean(x)
    y_mean = tf.reduce_mean(y)
    x_std_var = std_variance(x)
    y_std_var = std_variance(y)
    xy_covar = covariance(x,y)

    L = (2*x_mean*y_mean+c1) / (x_mean**2+y_mean**2+c1)   #lightness metric
    C = (2*x_std_var*y_std_var+c2) / (x_std_var**2+y_std_var**2+c2)   #contrast metric
    S = (xy_covar+c3) / (x_std_var*y_std_var+c3)   #structure metric
    SSIM = L*C*S
    # SSIM = ((2*x_mean*y_mean+c1)*(2*xy_covar+c2))/((x_mean**2+y_mean**2+c1)*(x_std_var**2+y_std_var**2+c2))

    loss = 1-SSIM

    return L, C, S, loss
