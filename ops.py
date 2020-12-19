# -*- coding:utf-8 -*-

from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import tensorflow as tf

def bias(name, shape, bias_start=0.0, trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable,
                          initializer=tf.constant_initializer(
                              bias_start, dtype=dtype))
    return var


def weight(name, shape, stddev=0.02, trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable,
                          initializer=tf.random_normal_initializer(
                              stddev=stddev, dtype=dtype))
    return var


# 全连接层
def fully_connected(value, output_shape, name='fully_connected', with_w=False):
    shape = value.get_shape().as_list()

    with tf.variable_scope(name):
        weights = tf.get_variable('matrixs', [shape[1], output_shape],tf.float32, trainable=True,initializer=tf.random_normal_initializer(stddev=0.02, dtype=tf.float32))
        biases = tf.get_variable('biases', [output_shape], tf.float32, trainable=True,initializer=tf.random_normal_initializer(stddev=0.02, dtype=tf.float32))

        if with_w:
            return tf.matmul(value, weights) + biases, weights, biases
        else:
            return tf.matmul(value, weights) + biases


# Leaky-ReLu 层
def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        return tf.maximum(x, leak * x, name=name)


# ReLu 层
def relu(value, name='relu'):
    with tf.variable_scope(name):
        return tf.nn.relu(value)


# 解卷积层
def deconv2d(value, output_shape, k_h=5, k_w=5, strides=[1, 2, 2, 1],
             name='deconv2d', with_w=False):
    with tf.variable_scope(name):
        weights = tf.get_variable('weights',[k_h, k_w, output_shape[-1], value.get_shape()[-1]],tf.float32, trainable=True,initializer=tf.random_normal_initializer(stddev=0.02, dtype=tf.float32))
        deconv = tf.nn.conv2d_transpose(value, weights,
                                        output_shape, strides=strides)
        biases = tf.get_variable('biases', [output_shape[-1]],tf.float32, trainable=True,initializer=tf.random_normal_initializer(stddev=0.02, dtype=tf.float32))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, weights, biases
        else:
            return deconv


# 卷积层
def conv2d(value, output_dim, k_h=5, k_w=5,strides=[1, 2, 2, 1], name='conv2d'):
    with tf.variable_scope(name):
        weights = tf.get_variable('weights',
                         [k_h, k_w, value.get_shape()[-1], output_dim],tf.float32, trainable=True,initializer=tf.random_normal_initializer(stddev=0.02, dtype=tf.float32))
        conv = tf.nn.conv2d(value, weights, strides=strides, padding='SAME')
        biases = tf.get_variable('biases', [output_dim],tf.float32, trainable=True,initializer=tf.random_normal_initializer(stddev=0.02, dtype=tf.float32))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


# 把约束条件串联到 feature map
def conv_cond_concat(value, cond, name='concat'):
    # 把张量的维度形状转化成 Python 的 list
    value_shapes = value.get_shape().as_list()
    cond_shapes = cond.get_shape().as_list()

    # 在第三个维度上（feature map 维度上）把条件和输入串联起来，
    # 条件会被预先设为四维张量的形式，假设输入为 [64, 32, 32, 32] 维的张量，
    # 条件为 [64, 32, 32, 10] 维的张量，那么输出就是一个 [64, 32, 32, 42] 维张量

    with tf.variable_scope(name):
        return tf.concat( [value, cond * tf.ones(value_shapes[0:3] + cond_shapes[3:])],3)

# Batch Normalization 层

def batch_norm_layer(value, is_train=True, name='batch_norm'):
    with tf.variable_scope(name) as scope:
        if is_train:
            return batch_norm(value, decay=0.9, epsilon=1e-5, scale=True,
                              is_training=is_train,
                              updates_collections=None, scope=scope)
        else:
            return batch_norm(value, decay=0.9, epsilon=1e-5, scale=True,
                              is_training=is_train, reuse=True,
                              updates_collections=None, scope=scope)


