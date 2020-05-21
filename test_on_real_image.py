import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from ForkNet import srcnn_ete, LOSS
from utils.utils import dolp, psnr, view_bar, aop, pad_shift, count_para, plot_feature_map, fig2array, interpolate_bic
import imageio as imgio
import os
import math
import time
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.measure import compare_ssim

vis_feature_map = False
hot_map = True
plot_dir = './images/feature_maps/'
test_img_path = './data/real_mosaic_image/'
image_index = 1
model_path = './best_model/model_1/model_1.ckpt'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

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
    test_img = np.array(Image.open(test_img_path + 'car_%d.bmp'%image_index), np.float32) / 255.
    test_img_ = test_img.reshape(1, test_img.shape[0], test_img.shape[1], 1)
    S0_hat_test, DoLP_hat_test, AoP_hat_test = sess.run([S0_hat, DoLP_hat, AoP_hat], feed_dict={Y: test_img_})

    S0_hat_test = np.clip(S0_hat_test[0, :, :, 0], 0, 2)
    DoLP_hat_test = np.clip(DoLP_hat_test[0, :, :, 0], 0, 1)
    AoP_hat_test = np.clip(AoP_hat_test[0, :, :, 0], 0, math.pi/2)

    imgio.imsave("./images/pred_s0_real_{}.bmp".format(image_index), S0_hat_test)
    imgio.imsave("./images/pred_dolp_real_{}.bmp".format(image_index), DoLP_hat_test)
    imgio.imsave("./images/pred_aop_real_{}.bmp".format(image_index), AoP_hat_test)

    bic_img,_ = interpolate_bic(test_img)
    S0_BIC = (bic_img[:, :, 0] + bic_img[:, :, 1] + bic_img[:, :, 2] + bic_img[:, :, 3]) / 2.
    DoLP_BIC = dolp(bic_img[:, :, 0], bic_img[:, :, 1], bic_img[:, :, 2], bic_img[:, :, 3])
    AoP_BIC = aop(bic_img[:, :, 0], bic_img[:, :, 1], bic_img[:, :, 2], bic_img[:, :, 3]) + math.pi / 4.

    imgio.imsave("./images/bic_s0_real_{}.bmp".format(image_index), S0_BIC)
    imgio.imsave("./images/bic_dolp_real_{}.bmp".format(image_index), DoLP_BIC)
    imgio.imsave("./images/bic_aop_real_{}.bmp".format(image_index), AoP_BIC)

    if hot_map:
        plt.axis('off')
        fig = plt.figure(1)
        fig = plt.gcf()
        height, width = AoP_hat_test.shape
        fig.set_size_inches(width/300, height/300)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)

        plt.imshow(AoP_hat_test, cmap=cm.jet)
        plt.savefig('./images/pred_aop_real_hotmap_{}.jpg'.format(image_index), dpi=300)
        plt.close(fig)


        plt.axis('off')
        fig = plt.figure(2)
        fig = plt.gcf()
        height, width = AoP_BIC.shape
        fig.set_size_inches(width / 300, height / 300)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)

        plt.imshow(AoP_BIC, cmap=cm.jet)
        plt.savefig('./images/bic_aop_real_hotmap_{}.jpg'.format(image_index), dpi=300)
        plt.close(fig)

