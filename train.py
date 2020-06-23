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
import h5py
from model import ForkNet, MSE_LOSS, LOSS, smooth_loss
from utils.batch_generator import patch_batch_generator
from utils.utils import dolp, psnr, normalize, aop, gs_rand_choice
import matplotlib.pyplot as plt 
import os
import math
import csv
from skimage.measure import compare_ssim

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#FINE_TUNE = False
LEARNING_RATE = 0.001
LEARNING_RATE_DECAY_STEPS = 600
LEARNING_RATE_DECAY_RATE = 0.988
IMG_NUM = 110
EPOCH_NUM = 200
BATCH_SIZE = 128
PATCH_WIDTH = 40
PATCH_HEIGHT = 40
GPUS = "2"
DSP_ITV = 6
metrics = 'training loss'
save_best = True
early_stop = False
patient = 5
train_img_index_path = './list/train_image_index.list'
val_img_index_path = './list/val_image_index.list'
Y_path = './data/training_set/Y.h5'
labels_path = './data/training_set/Labels.h5'
BIC_path = './data/training_set/BIC.h5'
ckpt_path = './best_model/model_1/model_1.ckpt'
csv_path = './list/psnr_record_1.csv'

#------------------------------------------------------------------------------
def load_data(batch_size = BATCH_SIZE, train_img_index_path = train_img_index_path,
              val_img_index_path = val_img_index_path, Y_path = Y_path, labels_path = labels_path):
    '''
    Divide the training set and validation set.
    Return two generator to generate batches of data.
    '''
    # read data from h5 file
    with h5py.File(Y_path, 'r') as h1:
        Y = np.array(h1.get('inputs'))

    with h5py.File(labels_path, 'r') as h2:
        label = np.array(h2.get('labels'))

    with h5py.File(BIC_path, 'r') as h3:
        bic = np.array(h3.get('bic'))

    Input = np.concatenate((Y, bic), axis=-1)

    patch_num = Y.shape[0]
    patch_num_per_img = patch_num // IMG_NUM

    train_img_index_str = open(train_img_index_path).read()
    train_img_index = [int(idx) for idx in train_img_index_str.split(',')]
    # train_img_num = len(train_img_index)

    val_img_index_str = open(val_img_index_path).read()
    val_img_index = [int(idx) for idx in val_img_index_str.split(',')]
    # val_img_num = len(val_img_index)

    patch_index_train = np.concatenate(
        [np.arange(i * patch_num_per_img, (i + 1) * patch_num_per_img) for i in train_img_index])
    patch_index_val = np.concatenate(
        [np.arange(i * patch_num_per_img, (i + 1) * patch_num_per_img) for i in val_img_index])

    # patch_index_train = np.concatenate(
    #     [gs_rand_choice(i * patch_num_per_img, (i + 1) * patch_num_per_img, 192) for i in train_img_index])
    # patch_index_val = np.concatenate(
    #     [gs_rand_choice(i * patch_num_per_img, (i + 1) * patch_num_per_img, 192) for i in val_img_index])

    # patch_index_train = np.concatenate(
    #     [np.random.choice(np.arange(i * patch_num_per_img, (i + 1) * patch_num_per_img), 192) for i in train_img_index])
    # patch_index_val = np.concatenate(
    #     [np.random.choice(np.arange(i * patch_num_per_img, (i + 1) * patch_num_per_img), 192) for i in val_img_index])

    patch_num_train = len(patch_index_train)
    # training steps in one epoch
    train_steps = int(np.ceil(patch_num_train * 1. / batch_size))
    print('# Training Patches: {}.'.format(patch_num_train))

    patch_num_val = len(patch_index_val)
    # validation steps in one epoch
    val_steps = int(np.ceil(patch_num_val * 1. / batch_size))
    print('# Validation Patches: {}.'.format(patch_num_val))

    train_Y = Input[patch_index_train]
    train_label = label[patch_index_train]
    val_Y = Input[patch_index_val]
    val_para = label[patch_index_val]
    # val_bic = bic[patch_index_val]

    return train_steps, train_Y, train_label, val_steps, val_Y, val_para

#------------------------------------------------------------------------------
def train(patch_width = PATCH_WIDTH, patch_height = PATCH_HEIGHT, epoch_num = EPOCH_NUM, batch_size = BATCH_SIZE,  learning_rate = LEARNING_RATE,
          learning_rate_decay_steps = LEARNING_RATE_DECAY_STEPS, learning_rate_decay_rate = LEARNING_RATE_DECAY_RATE,
          dsp_itv = DSP_ITV, ckpt_path = ckpt_path, save_best = save_best, early_stop = early_stop):
    '''
    Difine the tensorflow graph, execute training and validation.
    '''
    #------------------------------define the graph---------------------------
    Y = tf.placeholder(tf.float32, [None, None, None, 1], name='Y')
    S0 = tf.placeholder(tf.float32, [None, None, None, 1], name='S0')
    DoLP = tf.placeholder(tf.float32, [None, None, None, 1], name='DoLP')
    AoP = tf.placeholder(tf.float32, [None, None, None, 1], name='AoP')
#    Para = tf.placeholder(tf.float32, [None, None, None, 3])#define tensors of input and label
    
    # DoLP_hat= srcnn_ete(Y)
    S0_hat, DoLP_hat, AoP_hat = srcnn_ete(Y, padding='SAME')
    loss = LOSS(S0_hat, S0, DoLP_hat, DoLP, AoP_hat, AoP)
    tf.add_to_collection('S0_hat', S0_hat)
    tf.add_to_collection('DoLP_hat', DoLP_hat)
    tf.add_to_collection('AoP_hat', AoP_hat)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    decayed_learning_rate = tf.train.exponential_decay(learning_rate, global_step, learning_rate_decay_steps,
                                                       learning_rate_decay_rate, staircase=True)  # lerning rate decay
    
    train_step = tf.train.AdamOptimizer(decayed_learning_rate).minimize(loss, global_step = global_step)
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()  #instantiate the saveer
    
    train_steps, train_Y, train_label, val_steps, val_Y, val_label = load_data()
    # val_s0 = val_para[:, :, :, :1]
    # val_dolp = val_para[:, :, :, 1:2]
    # val_aop = val_para[:, :, :, 2:]
#    print(np.max(val_Y))

    with tf.Session() as sess:
        print('\nStart Training!')
        sess.run(init)
        min_loss = np.inf
        wait = 0
        psnr_record = []
        total_S0_PSNR_BIC = 0
        total_DoLP_PSNR_BIC = 0
        total_AoP_PSNR_BIC = 0

        for epoch in range(epoch_num):          
            #training set batch generator   
            train_generator = patch_batch_generator(train_Y, train_label, batch_size, patch_width, patch_height, random_shuffle = True)

             #test set batch generator   
            val_generator = patch_batch_generator(val_Y, val_label, batch_size, patch_width, patch_height, random_shuffle=False, augment=False)
            
            print('=======================================Epoch:{}/{}======================================='.format(epoch, epoch_num))
            # training
            total_train_loss = 0
            for step in range(train_steps):
                (Input_batch_train, Para_batch_train) = next(train_generator)

                Y_batch_train = Input_batch_train[:, :, :, :1]
                S0_batch_train = Para_batch_train[:,:,:,:1]
                DoLP_batch_train = Para_batch_train[:,:,:,1:2]
                AoP_batch_train = Para_batch_train[:,:,:,2:]

                sess.run(train_step, feed_dict={Y:Y_batch_train, S0:S0_batch_train, DoLP:DoLP_batch_train, AoP:AoP_batch_train})
                train_loss = sess.run(loss, feed_dict={Y:Y_batch_train, S0:S0_batch_train, DoLP:DoLP_batch_train, AoP:AoP_batch_train})
                total_train_loss += train_loss

                if step % dsp_itv == 0:
                    rate = float(step + 1) / float(train_steps)
                    rate_num = int(rate * 100)
                    arrow = 0 if step + 1 == train_steps else 1
                    r = '\rStep:%d/%d [%s%s%s]%d%% --- Training loss:%f' % \
                        (step + 1, train_steps, '■' * rate_num, '▶' * arrow, '-' * (100 - rate_num - arrow), rate * 100, train_loss)
                    print(r)
                
            # validation

            total_val_loss = 0
            total_S0_PSNR = 0
            total_DoLP_PSNR = 0
            total_AoP_PSNR = 0

            for step in range(val_steps):
                (Input_batch_val, Para_batch_val) = next(val_generator)

                Y_batch_val = Input_batch_val[:, :, :, :1]
                BIC_batch_val = Input_batch_val[:, :, :, 1:]
                S0_batch_val = Para_batch_val[:, :, :, :1]
                DoLP_batch_val = Para_batch_val[:, :, :, 1:2]
                AoP_batch_val = Para_batch_val[:, :, :, 2:]

                total_val_loss += sess.run(loss, feed_dict={Y: Y_batch_val, S0:S0_batch_val, DoLP:DoLP_batch_val, AoP:AoP_batch_val})
                S0_hat_val, DoLP_hat_val, AoP_hat_val = sess.run([S0_hat, DoLP_hat, AoP_hat], feed_dict={Y:Y_batch_val})
                #limit the value
                S0_hat_val = np.clip(S0_hat_val, 0, 2)
                DoLP_hat_val = np.clip(DoLP_hat_val, 0, 1)
                # AoP_hat_val = np.clip(AoP_hat_val, 0, math.pi)
                # DoLP_hat_val = Normalize(DoLP_hat_val, 0, 1)
                total_S0_PSNR += psnr(S0_batch_val[:, :, :, 0], S0_hat_val[:, :, :, 0], 2)
                total_DoLP_PSNR += psnr(DoLP_batch_val[:, :, :, 0], DoLP_hat_val[:, :, :, 0], 1)
                total_AoP_PSNR += psnr(AoP_batch_val[:, :, :, 0], AoP_hat_val[:, :, :, 0], math.pi / 2.)
                # for b in range(AoP_batch_val.shape[0]):
                #     total_AoP_PSNR += compare_ssim(np.float32(AoP_batch_val[b, :, :, 0]), np.float32(AoP_hat_val[b, :, :, 0]), data_range=math.pi / 2.)

                if epoch == 0:
    #                print('max:', max(val_bic[0,6:-6,6:-6,0]))
                    S0_BIC = (BIC_batch_val[:,:,:,0] + BIC_batch_val[:,:,:,1] + BIC_batch_val[:,:,:,2] + BIC_batch_val[:,:,:,3]) / 2.
                    DoLP_BIC = dolp(BIC_batch_val[:,:,:,0], BIC_batch_val[:,:,:,1], BIC_batch_val[:,:,:,2], BIC_batch_val[:,:,:,3])
                    AoP_BIC = aop(BIC_batch_val[:,:,:,0], BIC_batch_val[:,:,:,1], BIC_batch_val[:,:,:,2], BIC_batch_val[:,:,:,3]) + math.pi / 4.  #avoid the minus number
                    total_S0_PSNR_BIC += psnr(S0_batch_val[:, :, :, 0], S0_BIC, 2)
                    total_DoLP_PSNR_BIC += psnr(DoLP_batch_val[:, :, :, 0], DoLP_BIC, 1)
                    total_AoP_PSNR_BIC += psnr(AoP_batch_val[:, :, :, 0], AoP_BIC, math.pi / 2.)

                    # for b in range(AoP_batch_val.shape[0]):
                    #     total_AoP_PSNR_BIC += compare_ssim(np.float32(AoP_batch_val[b, :, :, 0]), np.float32(AoP_BIC[b]), data_range=math.pi / 2.)
                
            print('========================================Validation=======================================' +
                  '\nTraining loss: %.5f' % (total_train_loss/train_steps) + 
                  '\nValidation loss: %.5f' % (total_val_loss/val_steps) +
                  '\n ————————————————————————————————————————————————————————————————————————————————' + 
#                  '\n| PSNR of I_0 using PDCNN: %.5f    |   PSNR of I_0 using BICUBIC: %.5f   |' % (total_X_0_PSNR/val_steps, 32.872) + 
#                  '\n| PSNR of I_45 using PDCNN: %.5f   |   PSNR of I_45 using BICUBIC: %.5f  |' % (total_X_45_PSNR/val_steps, 32.973) + 
#                  '\n| PSNR of I_90 using PDCNN: %.5f   |   PSNR of I_90 using BICUBIC: %.5f  |' % (total_X_90_PSNR/val_steps, 33.008) + 
#                  '\n| PSNR of I_135 using PDCNN: %.5f  |   PSNR of I_135 using BICUBIC: %.5f |' % (total_X_135_PSNR/val_steps, 32.923) + 
                  '\n| PSNR of S_0 using SRCNN: %.5f    |   PSNR of S_0 using BICUBIC: %.5f   |' % (total_S0_PSNR/val_steps, total_S0_PSNR_BIC/val_steps) +
                  '\n| PSNR of DoLP using SRCNN: %.5f   |   PSNR of DoLP using BICUBIC: %.5f  |' % (total_DoLP_PSNR/val_steps, total_DoLP_PSNR_BIC/val_steps) +
                  '\n| PSNR of AoP using SRCNN: %.5f    |   PSNR of AoP using BICUBIC: %.5f   |' % (total_AoP_PSNR/val_steps, total_AoP_PSNR_BIC/val_steps) +
                  '\n ————————————————————————————————————————————————————————————————————————————————')

            psnr_record.append([total_S0_PSNR / val_steps, total_DoLP_PSNR / val_steps, total_AoP_PSNR / val_steps])
            
            if save_best or early_stop:
                if metrics == 'validation loss':
                    current_loss = total_val_loss/val_steps
                elif metrics == 'training loss':
                    current_loss = total_train_loss/train_steps
                if current_loss < min_loss:      
                    print('Validation loss decreased from %.5f to %.5f' % (min_loss, current_loss))
                    min_loss = current_loss
                    if save_best:
                        saver.save(sess, ckpt_path)
                        print("Model saved in file: %s" % ckpt_path)
                    if early_stop:   
                        wait = 0
                else:
                    print('Validation loss did not decreased.')
                    if early_stop:   
                        wait += 1
                        if wait > patient:
                            print('Early stop!')
                            break                         
        if not save_best:
            saver.save(sess, ckpt_path)
            print("Model saved in file: %s" % ckpt_path)

        with open(csv_path,'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(psnr_record)
                       
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = GPUS
    tf.reset_default_graph()
    train()
    
