# -*- coding: utf-8 -*-
"""
========================================================================
ForkNet for DoFP sensor to reconstruct s0, dolp and aop, Version 1.0
Copyright(c) 2020 Xianglong Zeng, Yuan Luo, Xiaojing Zhao, Wenbin Ye
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
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from model import ForkNet, LOSS
import imageio as imgio
import os
import math
from utils.utils import fig2array, normalize, view_bar
import time
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.measure import compare_ssim

IMG_NUM = 10
IMG_WIDTH = 1280
IMG_HEIGHT = 960
output_images=[9]
vis_feature_map = False
hot_map = False
plot_dir = './images/feature_maps/'
video_path = './data/video_data/temp.avi'
image_index = 3
model_path = './best_model/model_2/model_2.ckpt'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

tf.reset_default_graph()

Y = tf.placeholder(tf.float32, [None, None, None, 1], name='Y')
S0 = tf.placeholder(tf.float32, [None, None, None, 1], name='S0')
DoLP = tf.placeholder(tf.float32, [None, None, None, 1], name='DoLP')
AoP = tf.placeholder(tf.float32, [None, None, None, 1], name='AoP')
#    Para = tf.placeholder(tf.float32, [None, None, None, 3])#define tensors of input and label

# DoLP_hat= ForkNet(Y)
S0_hat, DoLP_hat, AoP_hat = ForkNet(Y, padding='SAME')

with tf.Session() as sess:
    saver = tf.train.Saver(var_list=tf.global_variables())
    saver.restore(sess,model_path)
    test_video = cv2.VideoCapture(video_path)
    size = (int(test_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(test_video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = test_video.get(cv2.CAP_PROP_FPS)
    total_frame = test_video.get(cv2.CAP_PROP_FRAME_COUNT)
    cur_frame = 1
    print('Video size:', size)
    # instantialize the writer
    s0_writer = cv2.VideoWriter('./data/video_data/s0.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, size)
    dolp_writer = cv2.VideoWriter('./data/video_data/dolp.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, size)
    # writer = cv2.VideoWriter('./data/video_data/aop.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, size)
    while(test_video.isOpened()):
        reading, frame = test_video.read()
        if not reading:
            break

        frame = np.dot(frame, [0.2989, 0.5870, 0.1140])
        frame = frame[None, :, :, None] / 255.
        S0_hat_test, DoLP_hat_test, AoP_hat_test = sess.run([S0_hat, DoLP_hat, AoP_hat], feed_dict={Y: frame})

        S0_hat_test = np.clip(S0_hat_test[0, :, :, 0], 0, 2)
        DoLP_hat_test = np.clip(DoLP_hat_test[0, :, :, 0], 0, 1)
        AoP_hat_test = np.clip(AoP_hat_test[0, :, :, 0], 0, math.pi)

        if hot_map:
            plt.axis('off')
            fig = plt.figure()
            fig = plt.gcf()
            height, width = AoP_hat_test.shape
            fig.set_size_inches(width/300, height/300)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.imshow(AoP_hat_test, cmap=cm.jet)
            hm = fig2array(fig)

        S0_hat_test = np.uint8(normalize(S0_hat_test, 0, 255))
        DoLP_hat_test = np.uint8(normalize(DoLP_hat_test, 0, 255))

        S0_hat_test = np.stack([S0_hat_test] *3 , axis=-1)
        DoLP_hat_test = np.stack([DoLP_hat_test] * 3, axis=-1)

        s0_writer.write(S0_hat_test)
        dolp_writer.write(DoLP_hat_test)

        view_bar(cur_frame-1, total_frame)
        cur_frame += 1

    test_video.release()
    s0_writer.release()
    dolp_writer.release()
    cv2.destroyAllWindows()




