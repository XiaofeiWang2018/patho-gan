# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import math


e = math.e
thre=0.5

def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)



class AlexNet (object):



    def __init__(self,input,num_classes,is_training = True,dropout_keep_prob = 0.5,spatial_squeeze = True):
        self.inputs=input
        self.num_classes=num_classes
        self.is_training = is_training
        self.dropout_keep_prob = dropout_keep_prob
        self.spatial_squeeze = spatial_squeeze  # 参数标志是否对输出进行squeeze操作（去除维度数为1的维度，比如5*3*1转为5*3）
        self.scope = 'alexnet'
        self.trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

    def alexnet_v2_arg_scope(weight_decay=0.0005):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,  # relu是AlexNet的一大亮点，取代了之前的softmax。
                            biases_initializer=tf.constant_initializer(0.1),
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.conv2d], padding='SAME'):  # 一般卷积的padding模式为SAME
                with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:  # pool的padding模式为VALID
                    return arg_sc

    def alexnet_v2(self):
        """ conv层降维"""
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
            net1 = slim.conv2d(self.inputs, 64, [11, 11], 4, padding='VALID',scope='conv1') #55

            net2 = slim.max_pool2d(net1, [3, 3], 2, scope='pool1') #27
            net3 = slim.conv2d(net2, 192, [5, 5], scope='conv2')  #27
            net4 = slim.max_pool2d(net3, [3, 3], 2, scope='pool2') #13
            net5 = slim.conv2d(net4, 384, [3, 3], scope='conv3') #13
            net6 = slim.conv2d(net5, 384, [3, 3], scope='conv4') #13
            net7 = slim.conv2d(net6, 256, [3, 3], scope='conv5') #13
            net8 = slim.max_pool2d(net7, [3, 3], 2, scope='pool5') #6
            # Use conv2d instead of fully_connected layers.
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=self.trunc_normal(0.005),
                                biases_initializer=tf.constant_initializer(0.1)):
                net9 = slim.conv2d(net8, 4096, [5, 5], padding='VALID',
                                  scope='fc6')
                net10 = slim.dropout(net9, self.dropout_keep_prob, is_training=self.is_training,
                                   scope='dropout6')
                net11 = slim.conv2d(net10, 4096, [2, 2], padding='VALID', scope='fc7')
                net12 = slim.dropout(net11, self.dropout_keep_prob, is_training=self.is_training,
                                   scope='dropout7')
                net13 = slim.conv2d(net12, self.num_classes, [1, 1],
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  biases_initializer=tf.zeros_initializer(),
                                  scope='fc8')

            # Convert end_points_collection into a end_point dict.

            if self.spatial_squeeze:
                net14 = tf.squeeze(net13, [1, 2], name='fc8/squeezed')  # 见后文详细注释

            return [net1,net2,net3,net4,net5,net6,net7,net8,net9,net10,net11,net12,net13,net14]

    def alexnet_v3(self):

        """ fc层降维"""
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
            net1 = slim.conv2d(self.inputs, 64, [11, 11], 4, padding='VALID',scope='conv1') #55
            net2 = slim.max_pool2d(net1, [3, 3], 2, scope='pool1') #27
            net3 = slim.conv2d(net2, 192, [5, 5], scope='conv2')  #27
            net4 = slim.max_pool2d(net3, [3, 3], 2, scope='pool2') #13
            net5 = slim.conv2d(net4, 384, [3, 3], scope='conv3') #13
            net6 = slim.conv2d(net5, 384, [3, 3], scope='conv4') #13
            net7 = slim.conv2d(net6, 256, [3, 3], scope='conv5') #13
            net8 = slim.max_pool2d(net7, [3, 3], 2, scope='pool5') #6
            # Use conv2d instead of fully_connected layers.

            with slim.arg_scope([slim.fully_connected],
                                weights_initializer=self.trunc_normal(0.005),
                                biases_initializer=tf.constant_initializer(0.1)):
                conv_shape = net8.get_shape().as_list()
                net9=tf.reshape(net8,shape=[-1,conv_shape[1]*conv_shape[2]*conv_shape[3]])
                net10 = slim.fully_connected(net9, 256, scope='fc6')
                net11 = slim.dropout(net10, self.dropout_keep_prob, is_training=self.is_training,scope='dropout6')
                net14 = slim.fully_connected(net11, self.num_classes,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  biases_initializer=tf.zeros_initializer(),
                                  scope='fc8')

            return [net1,net2,net3,net4,net5,net6,net7,net8,net9,net10,net11,net14]






