# -*- coding: utf-8 -*-
# Implementation of Wang et al 2017: Automatic Brain Tumor Segmentation using Cascaded Anisotropic Convolutional Neural Networks. https://arxiv.org/abs/1709.00382

# Author: Guotai Wang
# Copyright (c) 2017-2018 University College London, United Kingdom. All rights reserved.
# http://cmictig.cs.ucl.ac.uk
#
# Distributed under the BSD-3 licence. Please see the file licence.txt
# This software is not certified for clinical use.

from __future__ import absolute_import, print_function
import sys
import numpy as np
from scipy import ndimage
import time
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
from util.data_loader import *
from util.data_process import *
from util.train_test_func import *
from util.parse_config import parse_config
from train import NetFactory

def test(config_file):
    # 1, load configure file
    config = parse_config(config_file)
    config_data = config['data']
    config_net1 = config.get('network1', None)
    config_net2 = config.get('network2', None)
    config_net3 = config.get('network3', None)
    config_test = config['testing']  
    batch_size  = config_test.get('batch_size', 5)
    
    # 2.1, network for whole tumor
    if(config_net1):
        net_type1    = config_net1['net_type']
        net_name1    = config_net1['net_name']
        data_shape1  = config_net1['data_shape']
        label_shape1 = config_net1['label_shape']
        class_num1   = config_net1['class_num']
        
        # construct graph for 1st network
        full_data_shape1 = [batch_size] + data_shape1
        x1 = tf.placeholder(tf.float32, shape = full_data_shape1)          
        net_class1 = NetFactory.create(net_type1)
        net1 = net_class1(num_classes = class_num1,w_regularizer = None,
                    b_regularizer = None, name = net_name1)
        net1.set_params(config_net1)
        predicty1 = net1(x1, is_training = True)
        proby1 = tf.nn.softmax(predicty1)
    else:
        config_net1ax = config['network1ax']
        config_net1sg = config['network1sg']
        config_net1cr = config['network1cr']
        
        # construct graph for 1st network axial
        net_type1ax    = config_net1ax['net_type']
        net_name1ax    = config_net1ax['net_name']
        data_shape1ax  = config_net1ax['data_shape']
        label_shape1ax = config_net1ax['label_shape']
        class_num1ax   = config_net1ax['class_num']
        
        full_data_shape1ax = [batch_size] + data_shape1ax
        x1ax = tf.placeholder(tf.float32, shape = full_data_shape1ax)          
        net_class1ax = NetFactory.create(net_type1ax)
        net1ax = net_class1ax(num_classes = class_num1ax,w_regularizer = None,
                    b_regularizer = None, name = net_name1ax)
        net1ax.set_params(config_net1ax)
        predicty1ax = net1ax(x1ax, is_training = True)
        proby1ax = tf.nn.softmax(predicty1ax)

        # construct graph for 1st network sagittal
        net_type1sg    = config_net1sg['net_type']
        net_name1sg    = config_net1sg['net_name']
        data_shape1sg  = config_net1sg['data_shape']
        label_shape1sg = config_net1sg['label_shape']
        class_num1sg   = config_net1sg['class_num']

        full_data_shape1sg = [batch_size] + data_shape1sg
        x1sg = tf.placeholder(tf.float32, shape = full_data_shape1sg)          
        net_class1sg = NetFactory.create(net_type1sg)
        net1sg = net_class1sg(num_classes = class_num1sg,w_regularizer = None,
                    b_regularizer = None, name = net_name1sg)
        net1sg.set_params(config_net1sg)
        predicty1sg = net1sg(x1sg, is_training = True)
        proby1sg = tf.nn.softmax(predicty1sg)
            
        # construct graph for 1st network corogal
        net_type1cr    = config_net1cr['net_type']
        net_name1cr    = config_net1cr['net_name']
        data_shape1cr  = config_net1cr['data_shape']
        label_shape1cr = config_net1cr['label_shape']
        class_num1cr   = config_net1cr['class_num']

        full_data_shape1cr = [batch_size] + data_shape1cr
        x1cr = tf.placeholder(tf.float32, shape = full_data_shape1cr)          
        net_class1cr = NetFactory.create(net_type1cr)
        net1cr = net_class1cr(num_classes = class_num1cr,w_regularizer = None,
                    b_regularizer = None, name = net_name1cr)
        net1cr.set_params(config_net1cr)
        predicty1cr = net1cr(x1cr, is_training = True)
        proby1cr = tf.nn.softmax(predicty1cr)
 
    # 3, create session and load trained models
    print('create session and load trained models /n')
    model_t0 = time.time()

    # with tf.device("/device:GPU:0"): #0806
    all_vars = tf.global_variables()
    sess = tf.InteractiveSession()   
    sess.run(tf.global_variables_initializer())  
    if(config_net1):
        net1_vars = [x for x in all_vars if x.name[0:len(net_name1) + 1]==net_name1 + '/']
        saver1 = tf.train.Saver(net1_vars)
        saver1.restore(sess, config_net1['model_file'])
    else:
        net1ax_vars = [x for x in all_vars if x.name[0:len(net_name1ax) + 1]==net_name1ax + '/']
        saver1ax = tf.train.Saver(net1ax_vars)
        saver1ax.restore(sess, config_net1ax['model_file'])
        net1sg_vars = [x for x in all_vars if x.name[0:len(net_name1sg) + 1]==net_name1sg + '/']
        saver1sg = tf.train.Saver(net1sg_vars)
        saver1sg.restore(sess, config_net1sg['model_file'])     
        net1cr_vars = [x for x in all_vars if x.name[0:len(net_name1cr) + 1]==net_name1cr + '/']
        saver1cr = tf.train.Saver(net1cr_vars)
        saver1cr.restore(sess, config_net1cr['model_file'])

    print('Model load time is {}'.format(time.time() - model_t0))

    # 4, load test images
    print('load test images \n')
    load_t0 = time.time()
    dataloader = DataLoader(config_data)
    dataloader.load_data()
    image_num = dataloader.get_total_image_number()
    print('data load time is {}'.format(time.time()-load_t0))

    # 5, start to test
    print('start to test \n')
    test_slice_direction = config_test.get('test_slice_direction', 'all')
    # save_folder = config_data['save_folder']
    #Ben:save the segment output to the same folder as the input
    save_folder = '.'
    test_time = []
    struct = ndimage.generate_binary_structure(3, 2)
    margin = config_test.get('roi_patch_margin', 5)
    
    for i in range(image_num):
        [temp_imgs, temp_weight, temp_name, img_names, temp_bbox, temp_size] = dataloader.get_image_data_with_name(i)
        print(f'Segmenting on case {temp_name}\n')
        t0 = time.time()
        # 5.1, test of 1st network
        if(config_net1):
            data_shapes  = [ data_shape1[:-1],  data_shape1[:-1],  data_shape1[:-1]] 
            label_shapes = [label_shape1[:-1], label_shape1[:-1], label_shape1[:-1]]
            nets = [net1, net1, net1]
            outputs = [proby1, proby1, proby1]
            inputs =  [x1, x1, x1]
            class_num = class_num1
        else:
            data_shapes  = [ data_shape1ax[:-1],  data_shape1sg[:-1],  data_shape1cr[:-1]]
            label_shapes = [label_shape1ax[:-1], label_shape1sg[:-1], label_shape1cr[:-1]]
            nets = [net1ax, net1sg, net1cr]
            outputs = [proby1ax, proby1sg, proby1cr]
            inputs =  [x1ax, x1sg, x1cr]
            class_num = class_num1ax
        prob1 = test_one_image_three_nets_adaptive_shape(temp_imgs, data_shapes, label_shapes, data_shape1ax[-1], class_num,
                   batch_size, sess, nets, outputs, inputs, shape_mode = 2) #average probability of ax,sg,co
        pred1 =  np.asarray(np.argmax(prob1, axis = 3), np.uint16)
        pred1 = pred1 * temp_weight  #what is the temp_weight

        wt_threshold = 2000
        pred1_lc = ndimage.morphology.binary_closing(pred1, structure = struct)
        pred1_lc = get_largest_two_component(pred1_lc, False, wt_threshold)
        out_label = pred1_lc

        test_time.append(time.time() - t0)
        final_label = np.zeros(temp_size, np.int16)
        final_label = set_ND_volume_roi_with_bounding_box_range(final_label, temp_bbox[0], temp_bbox[1], out_label)
        #Todo check save path's existence, if not, mkdir
        subfolder = f'{save_folder}/{temp_name}'
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        save_array_as_nifty_volume(final_label, subfolder+"/{}_brain.nii.gz".format(temp_name.split('/')[-1]), img_names[0])

    test_time = np.asarray(test_time)
    print('test time', test_time.mean())
    np.savetxt(save_folder + '/test_time.txt', test_time)
    sess.close()
      
if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2,i.e config file path and data path, e.g.')
        print('python test.py config/test_all_class.txt')
        exit()
    config_file = str(sys.argv[1])
    assert (os.path.isfile(config_file))
    test(config_file)

