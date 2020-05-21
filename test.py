# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 23:01:08 2018

@author: Dell
"""
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from ForkNet import srcnn_ete, LOSS
from utils.utils import dolp, psnr, normalize, view_bar, aop, pad_shift, count_para, plot_feature_map, fig2array
import imageio as imgio
import os
import math
import time
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.measure import compare_ssim

IMG_NUM = 10
IMG_WIDTH = 1280
IMG_HEIGHT = 960
output_images=[]
vis_feature_map = False
hot_map = True
plot_dir = './images/feature_maps/'
test_img_path = './data/test_set'
model_path = './best_model/model_1/model_1.ckpt'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

bic_img = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH, 4], np.float32)
origin_img = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH, 4], np.float32)
msc_img = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH, 1], np.float32)
bic_s0 = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)
origin_s0 = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)
pred_s0 = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)
bic_dolp = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)
origin_dolp = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)
pred_dolp = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)
bic_aop = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)
origin_aop = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)
pred_aop = np.zeros([IMG_NUM, IMG_HEIGHT, IMG_WIDTH], np.float32)

# define the index for downsampling
m_1 = np.arange(0, 959, 2)
n_1 = np.arange(0, 1279, 2)
m_2 = np.arange(1, 960, 2)
n_2 = np.arange(1, 1280, 2)
i_0, j_0 = np.meshgrid(m_1, n_1, indexing='ij')
i_45, j_45 = np.meshgrid(m_1, n_2, indexing='ij')
i_90, j_90 = np.meshgrid(m_2, n_1, indexing='ij')
i_135, j_135 = np.meshgrid(m_2, n_2, indexing='ij')
ds_index = [(i_0, j_0), (i_45, j_45), (i_90, j_90), (i_135, j_135)]

tf.reset_default_graph()

Y = tf.placeholder(tf.float32, [None, None, None, 1], name='Y')
S0 = tf.placeholder(tf.float32, [None, None, None, 1], name='S0')
DoLP = tf.placeholder(tf.float32, [None, None, None, 1], name='DoLP')
AoP = tf.placeholder(tf.float32, [None, None, None, 1], name='AoP')
#    Para = tf.placeholder(tf.float32, [None, None, None, 3])#define tensors of input and label

# DoLP_hat= srcnn_ete(Y)
S0_hat, DoLP_hat, AoP_hat = srcnn_ete(Y, padding='SAME')

with tf.Session() as sess:
    saver = tf.train.Saver(var_list=tf.global_variables())
    saver.restore(sess,model_path)

    count_para()

    total_S0_PSNR = np.zeros((IMG_NUM))
    total_S0_PSNR_BIC = np.zeros((IMG_NUM))
    total_DoLP_PSNR = np.zeros((IMG_NUM))
    total_DoLP_PSNR_BIC = np.zeros((IMG_NUM))
    total_AoP_PSNR = np.zeros((IMG_NUM))
    total_AoP_PSNR_BIC = np.zeros((IMG_NUM))
    total_time = 0

    for i in range(0, IMG_NUM):
        for j in range(0, 4):
            path_origin = test_img_path + '/image_{}_{}.bmp'.format(i + 1, j * 45)
            img = np.array(Image.open(path_origin), np.float32) / 255.
            msc_img[i, ds_index[j][0], ds_index[j][1], 0] = img[ds_index[j]]
            bic_img[i, :, :, j] = cv2.resize(img[ds_index[j]], (IMG_WIDTH, IMG_HEIGHT), cv2.INTER_CUBIC)
            origin_img[i, :, :, j] = img

        bic_img[i]=pad_shift(bic_img[i])

        start = time.time()
        S0_hat_test, DoLP_hat_test, AoP_hat_test = sess.run([S0_hat, DoLP_hat, AoP_hat], feed_dict={Y: msc_img[i:i+1]})
        end = time.time()
        total_time += end - start

        # the s0, dolp and aop of predict images
        S0_hat_test = np.clip(S0_hat_test[0, :, :, 0], 0, 2)
        DoLP_hat_test = np.clip(DoLP_hat_test[0, :, :, 0], 0, 1)
        # AoP_hat_test = AoP_hat_test[0, :, :, 0]
        # print(np.max(AoP_hat_test), np.min(AoP_hat_test))
        AoP_hat_test = np.clip(AoP_hat_test[0, :, :, 0], 0, math.pi/2)
        # AoP_hat_test = Normalize(AoP_hat_test[0, :, :, 0], math.pi / 2., 0)
        pred_s0[i] = S0_hat_test
        pred_dolp[i] = DoLP_hat_test
        pred_aop[i] = AoP_hat_test

        #the s0, dolp and aop of original images
        S0_true = 0.5 * (origin_img[i, :, :, 0] + origin_img[i, :, :, 1] + origin_img[i, :, :, 2] + origin_img[i, :, :, 3])
        DoLP_true = dolp(origin_img[i, :, :, 0], origin_img[i, :, :, 1], origin_img[i, :, :, 2], origin_img[i, :, :, 3])
        AoP_true = aop(origin_img[i, :, :, 0], origin_img[i, :, :, 1], origin_img[i, :, :, 2], origin_img[i, :, :, 3]) + math.pi/4.
        origin_s0[i] = S0_true
        origin_dolp[i] = DoLP_true
        origin_aop[i] = AoP_true

        #the dolp of bic images
        S0_BIC = 1 / 2 * (bic_img[i, :, :, 0] + bic_img[i, :, :, 1] + bic_img[i, :, :, 2] + bic_img[i,:, :, 3])
        DoLP_BIC = dolp(bic_img[i, :, :, 0], bic_img[i, :, :, 1], bic_img[i, :, :, 2], bic_img[i, :, :, 3])
        AoP_BIC = aop(bic_img[i, :, :, 0], bic_img[i, :, :, 1], bic_img[i, :, :, 2], bic_img[i, :, :, 3]) + math.pi / 4.
        bic_s0[i] = S0_BIC
        bic_dolp[i] = DoLP_BIC
        bic_aop[i] = AoP_BIC

        # Calculate the PSNR of S0, DoLP and AoP obtained through PDCNN method
        total_S0_PSNR[i]=  psnr(S0_true, S0_hat_test, 2)
        total_DoLP_PSNR[i] = psnr(DoLP_true, DoLP_hat_test, 1)
        total_AoP_PSNR[i] = psnr(AoP_true, AoP_hat_test, math.pi/2.)

        # Calculate the PSNR of S0, DoLP and AoP obtained through BICUBIC method
        total_S0_PSNR_BIC[i] = psnr(S0_true, S0_BIC, 2)
        total_DoLP_PSNR_BIC[i] = psnr(DoLP_true, DoLP_BIC, 1)
        total_AoP_PSNR_BIC[i] = psnr(AoP_true, AoP_BIC, math.pi/2.)

        # show the progress bar
        view_bar(i, IMG_NUM)

    print('\n========================================Testing=======================================' +
          '\n ————————————————————————————————————————————————————————————————————————————————' +
          '\n| PSNR of S_0 using SRCNN: %.5f    |   PSNR of S_0 using BICUBIC: %.5f   |' % (np.mean(total_S0_PSNR), np.mean(total_S0_PSNR_BIC)) +
          '\n| PSNR of DoLP using PDCNN: %.5f   |   PSNR of DoLP using BICUBIC: %.5f  |' % (np.mean(total_DoLP_PSNR), np.mean(total_DoLP_PSNR_BIC)) +
          '\n| PSNR of AoP using SRCNN: %.5f    |   PSNR of AoP using BICUBIC: %.5f   |' % (np.mean(total_AoP_PSNR), np.mean(total_AoP_PSNR_BIC)) +
          '\n ————————————————————————————————————————————————————————————————————————————————')

    print('\nSRCNN time: {} sec'.format(total_time / IMG_NUM))

    for j in output_images:

        imgio.imsave("./images/bic_s0_{}_{}.jpg".format(j, total_S0_PSNR_BIC[j-1]), bic_s0[j-1, 390:-470, 690:-490])
        imgio.imsave("./images/pred_s0_3_path_{}_{}.jpg".format(j, total_S0_PSNR[j-1]), pred_s0[j-1, 200:-80, 325:-275])
        imgio.imsave("./images/org_s0_{}.jpg".format(j), origin_s0[j-1, 390:-470, 690:-490])

        imgio.imsave("./images/bic_dolp_{}_{}.jpg".format(j, total_DoLP_PSNR_BIC[j - 1]), bic_dolp[j - 1, 285:-575, 865:-315])
        imgio.imsave("./images/pred_dolp_3_path_{}_{}.jpg".format(j, total_DoLP_PSNR[j - 1]), pred_dolp[j - 1, 200:-80, 325:-275])
        imgio.imsave("./images/org_dolp_{}.jpg".format(j), origin_dolp[j - 1, 285:-575, 865:-315])

        imgio.imsave("./images/bic_aop_{}_{}.jpg".format(j, total_AoP_PSNR_BIC[j - 1]), bic_aop[j - 1, 200:-80, 325:-275])
        imgio.imsave("./images/pred_aop_3_path_{}_{}.jpg".format(j, total_AoP_PSNR[j - 1]), pred_aop[j - 1, 200:-80, 325:-275])
        imgio.imsave("./images/org_aop_{}.jpg".format(j), origin_aop[j - 1, 200:-80, 325:-275])

        if hot_map:
            plt.axis('off')
            fig = plt.figure()
            fig = plt.gcf()
            height, width = bic_aop[j - 1, 200:-80, 325:-275].shape
            fig.set_size_inches(width/300, height/300)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)

            plt.imshow(bic_aop[j - 1, 200:-80, 325:-275], cmap=cm.jet)
            plt.savefig('./images/bic_aop.jpg', dpi=300)

            plt.imshow(pred_aop[j - 1, 200:-80, 325:-275],  cmap=cm.jet)
            plt.savefig('./images/pred_aop_0.05_0.0180_log.jpg', dpi=300)

            plt.imshow(origin_aop[j - 1, 200:-80, 325:-275], cmap=cm.jet)
            plt.savefig('./images/org_aop.jpg', dpi=300)

            # imgio.imsave("./images/msc_img_{}.jpg".format(j), msc_img[j - 1, 320:-320, 410:-410, :])

        if vis_feature_map:
            visualize_layers = ['x_1', 'x_2', 'x_3_1', 'x_3_2', 'x_3_3']
            conv_out = sess.run(tf.get_collection('feature_maps'), feed_dict={Y: msc_img[j-1:j]})
            for m, layer in enumerate(visualize_layers):
                if not os.path.exists(plot_dir + layer):
                    os.mkdir(plot_dir + layer)
                plot_feature_map(conv_out[m][:, 320:-320, 410:-410], plot_dir + layer, maps_all=True)

