# -*- coding: utf-8 -*-
import numpy as np
import scipy.misc
from alexnet import *
from matplotlib import pyplot as plt
from skimage import io, transform
from scipy.misc import imread, imresize
from data_processing import DataLoader_vessel as DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
L_ad = 1
L_content = 1
L_tv = 0
mode = 'Ours'
BATCH_SIZE = 2

sysstr = "Linux"
Z_DIM = 400
LR = 0.0002
LR_str = '_lr2-4'
dataset = 'OURS'
img_H = 512
save_size = [1, 1]

if not os.path.isdir('result/' + dataset + '/' + mode + '/Lad_' + str(L_ad) + '_Lst_' + str(L_content) + '_Ltv_' + str(
        L_tv) + '_uniform'):
    os.mkdir('result/' + dataset + '/' + mode + '/Lad_' + str(L_ad) + '_Lst_' + str(L_content) + '_Ltv_' + str(
        L_tv) + '_uniform')
SAVE_PATH = ('result/' + dataset + '/' + mode + '/Lad_' + str(L_ad) + '_Lst_' + str(L_content) + '_Ltv_' + str(
    L_tv) + '_uniform')
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True


def bp(loss_label, a, g, sess):
    with g.as_default():
        with g.gradient_override_map({'Relu': 'bpRelu'}):
            grads = tf.gradients(loss_label, a)[0]
    return grads


def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def save_images(images, size, path):
    """
    Save the samples images
    The best size number is
            int(max(sqrt(image.shape[0]),sqrt(image.shape[1]))) + 1
    example:
        The batch_size is 64, then the size is recommended [8, 8]
        The batch_size is 32, then the size is recommended [6, 6]
    """

    # 图片归一化，主要用于生成器输出是 tanh 形式的归一化
    img = images
    h, w = img.shape[1], img.shape[2]

    # 产生一个大画布，用来保存生成的 batch_size 个图像
    merge_img = np.zeros((h * size[0], w * size[1], 3))

    # 循环使得画布特定地方值为某一幅图像的值
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j * h:j * h + h, i * w:i * w + w, :] = image

    # 保存画布
    return scipy.misc.imsave(path, merge_img)


def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)


initializer = tf.truncated_normal_initializer(stddev=0.02)
bias_initializer = tf.constant_initializer(0.0)


def discriminator(image, reuse=False):
    n = 32
    bn = slim.batch_norm
    with tf.name_scope("disciminator"):
        # original
        dis1 = slim.convolution2d(image, n, [4, 4], 2, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv1', weights_initializer=initializer)

        dis2 = slim.convolution2d(dis1, 2 * n, [4, 4], 2, normalizer_fn=bn, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv2', weights_initializer=initializer)

        dis3 = slim.convolution2d(dis2, 4 * n, [4, 4], 2, normalizer_fn=bn, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv3', weights_initializer=initializer)

        dis4 = slim.convolution2d(dis3, 8 * n, [4, 4], 2, normalizer_fn=bn, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv4', weights_initializer=initializer)

        dis5 = slim.convolution2d(dis4, 16 * n, [4, 4], 2, normalizer_fn=bn, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv5', weights_initializer=initializer)

        dis6 = slim.convolution2d(dis5, 16 * n, [4, 4], 2, normalizer_fn=bn, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv6', weights_initializer=initializer)

        d_out_logits = slim.fully_connected(slim.flatten(dis6), 1, activation_fn=None, reuse=reuse, scope='d_out',
                                            weights_initializer=initializer)

        d_out = tf.nn.sigmoid(d_out_logits)
    return d_out, d_out_logits


def generator(image, z, n=64, is_train=True):
    with tf.name_scope("generator"):
        # original
        e1 = slim.conv2d(image, n, [4, 4], 2, activation_fn=lrelu, scope='g_e1_conv',
                         weights_initializer=initializer)
        # 256
        e2 = slim.conv2d(lrelu(e1), 2 * n, [4, 4], 2, normalizer_fn=slim.batch_norm, activation_fn=None,
                         scope='g_e2_conv',
                         weights_initializer=initializer)
        # 128
        e3 = slim.conv2d(lrelu(e2), 4 * n, [4, 4], 2, normalizer_fn=slim.batch_norm, activation_fn=None,
                         scope='g_e3_conv',
                         weights_initializer=initializer)
        # 64
        e4 = slim.conv2d(lrelu(e3), 8 * n, [4, 4], 2, normalizer_fn=slim.batch_norm, activation_fn=None,
                         scope='g_e4_conv',
                         weights_initializer=initializer)
        # 32
        e5 = slim.conv2d(lrelu(e4), 8 * n, [4, 4], 2, normalizer_fn=slim.batch_norm, activation_fn=None,
                         scope='g_e5_conv',
                         weights_initializer=initializer)
        # # 16
        e6 = slim.conv2d(lrelu(e5), 8 * n, [4, 4], 2, normalizer_fn=slim.batch_norm, activation_fn=None,
                         scope='g_e6_conv',
                         weights_initializer=initializer)

        zP = slim.fully_connected(z, 8 * 8 * n, normalizer_fn=None, activation_fn=lrelu, scope='g_project',
                                  weights_initializer=initializer)
        zCon = tf.reshape(zP, [-1, 8, 8, n])

        # gen1 = slim.conv2d_transpose(lrelu(zCon), 2 * n, [4, 4], 2, normalizer_fn=slim.batch_norm, activation_fn=None,
        #                              scope='g_dconv1', weights_initializer=initializer)
        # 8
        gen1 = tf.concat([zCon, e6], 3)

        gen2 = slim.conv2d_transpose(lrelu(gen1), 8 * n, [4, 4], 2, normalizer_fn=slim.batch_norm, activation_fn=None,
                                     scope='g_dconv2', weights_initializer=initializer)
        # 16
        gen2 = tf.concat([gen2, e5], 3)

        gen3 = slim.conv2d_transpose(lrelu(gen2), 4 * n, [4, 4], 2, normalizer_fn=slim.batch_norm, activation_fn=None,
                                     scope='g_dconv3', weights_initializer=initializer)
        gen3 = tf.concat([gen3, e4], 3)

        # 32
        gen6 = slim.conv2d_transpose(tf.nn.relu(gen3), 2 * n, [4, 4], 2, normalizer_fn=slim.batch_norm,
                                     activation_fn=None,
                                     scope='g_dconv6', weights_initializer=initializer)
        gen6 = tf.concat([gen6, e3], 3)

        # 64
        gen7 = slim.conv2d_transpose(tf.nn.relu(gen6), n, [4, 4], 2, normalizer_fn=slim.batch_norm,
                                     activation_fn=None,
                                     scope='g_dconv7', weights_initializer=initializer)
        gen7 = tf.concat([gen7, e2], 3)

        gen8 = slim.conv2d_transpose(tf.nn.relu(gen7), n, [4, 4], 2, normalizer_fn=slim.batch_norm,
                                     activation_fn=None,
                                     scope='g_dconv8', weights_initializer=initializer)
        gen8 = tf.concat([gen8, e1], 3)

        # 128
        gen_out = slim.conv2d_transpose(tf.nn.relu(gen8), 3, [4, 4], 2, activation_fn=tf.nn.sigmoid,
                                        scope='g_out', weights_initializer=initializer)
        gen_out_227 = tf.image.resize_images(gen_out, [227, 227])
    return gen_out, gen_out_227


def styleloss_RNFLD(syn, style_gram, weight_gram, sess):
    """

    :param syn: tf N,227,227,3
    :param style_gram: ndarray N,6*6,256
    :param weight_gram:  ndarray N,6*6,256
    :param sess:
    :return:
    """

    net_syn = AlexNet(syn, num_classes=2, is_training=False)
    with slim.arg_scope(AlexNet.alexnet_v2_arg_scope()):
        tf.get_variable_scope().reuse_variables()
        k_syn = net_syn.alexnet_v3()
        cnn_output_syn = k_syn[7]
        variables = tf.contrib.framework.get_variables_to_restore()[71:85]
        saver_syn = tf.train.Saver(variables)
        model_file1 = tf.train.latest_checkpoint('./ckpt3/')
        saver_syn.restore(sess, model_file1)

    cnn_output_syn = tf.reshape(cnn_output_syn, shape=[-1, cnn_output_syn._shape_as_list()[1]
                                                       * cnn_output_syn._shape_as_list()[2],
                                                       cnn_output_syn._shape_as_list()[3]])  # N,6*6,256

    syn_gram = tf.multiply(weight_gram, cnn_output_syn)
    style_loss = tf.reduce_mean(tf.square(syn_gram - style_gram))
    return style_loss


def get_tv_loss(img):
    x = tf.reduce_mean(tf.abs(img[:, 1:, :, :] - img[:, :-1, :, :]))
    y = tf.reduce_mean(tf.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    return x + y


def main():
    sess = tf.InteractiveSession()
    global_step = tf.Variable(0, name='global_step', trainable=False)

    images = tf.placeholder(tf.float32, [BATCH_SIZE, img_H, img_H, 3], name='real_images')
    z = tf.placeholder(tf.float32, [BATCH_SIZE, Z_DIM], name='z')
    vessel = tf.placeholder(tf.float32, [BATCH_SIZE, img_H, img_H, 3], name='vessel')
    style_gram = tf.placeholder(tf.float32, [BATCH_SIZE, None, None], name='style_gram')
    weight_gram = tf.placeholder(tf.float32, [BATCH_SIZE, None, None], name='weight_gram')
    X = tf.placeholder(tf.float32, [None, 227, 227, 3])  # 输入: MNIST数据图像为展开的向量

    G, G_227 = generator(vessel, z)
    images_ = tf.concat([images, vessel], 3)
    G_ = tf.concat([G, vessel], 3)
    D, D_logits = discriminator(images_)
    D_, D_logits_ = discriminator(G_, reuse=True)

    sess.run(tf.global_variables_initializer())

    net = AlexNet(X, num_classes=2, is_training=False)
    with slim.arg_scope(AlexNet.alexnet_v2_arg_scope()):
        k = net.alexnet_v3()
        logits = k[11]
        norm_grads = tf.gradients(logits[:, 1], k[7])[0]  # 55,55,64
    variables = tf.contrib.framework.get_variables_to_restore()[71:85]
    saver_syn = tf.train.Saver(variables)
    model_file1 = tf.train.latest_checkpoint('./ckpt3/')
    saver_syn.restore(sess, model_file1)

    "---------------------------------------------------------------"

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=tf.ones_like(D)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.zeros_like(D_)))
    d_loss = d_loss_real + d_loss_fake
    g_loss_style = styleloss_RNFLD(G_227, style_gram, weight_gram, sess=sess)
    g_loss_ad = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.ones_like(D_)))
    g_loss_tv = get_tv_loss(G)
    g_loss = L_ad * g_loss_ad + L_content * g_loss_style + L_tv * g_loss_tv

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    saver2 = tf.train.Saver(max_to_keep=10)

    d_optim = tf.train.GradientDescentOptimizer(LR).minimize(d_loss, var_list=d_vars, global_step=global_step)
    g_optim = tf.train.GradientDescentOptimizer(LR).minimize(g_loss, var_list=g_vars, global_step=global_step)

    dataloader = DataLoader(batch_size=BATCH_SIZE, input_size=np.array([img_H, img_H]), sysstr=sysstr, path=dataset)
    dataloader_test = DataLoader(batch_size=BATCH_SIZE, input_size=np.array([img_H, img_H]), sysstr=sysstr,
                                 path=dataset)
    num_batches = dataloader.num_batches
    sample_z = np.random.uniform(0, 1, size=(BATCH_SIZE, Z_DIM))
    _, batch_vessel_test, _ = dataloader_test.get_batch()
    count = 0
    for epoch in range(2400):
        for idx in range(num_batches):
            batch_images, batch_vessel, img_name = dataloader.get_batch()
            batch_z = np.random.uniform(0, 1, size=(BATCH_SIZE, Z_DIM))
            batch_images_227 = transform.resize(batch_images, [BATCH_SIZE, 227, 227])  # N,227,227,3

            cnn_out, norm_grads_1 = sess.run([k[7], norm_grads], feed_dict={X: batch_images_227})

            weights = np.mean(np.abs(norm_grads_1), axis=(1, 2))  # N,256
            weight_gram_temp = np.expand_dims(weights, axis=1)  # N,1,256
            weight_gram_temp1 = np.repeat(weight_gram_temp, 6 * 6, axis=1)  # N,6*6,256
            Style_gram = np.reshape(cnn_out, [-1, cnn_out.shape[1] * cnn_out.shape[2], cnn_out.shape[3]])  # N，6*6,256
            style_gram1 = np.multiply(weight_gram_temp1, Style_gram)

            feed_dict_g = {images: batch_images, z: batch_z, vessel: batch_vessel, weight_gram: weight_gram_temp1,
                           style_gram: style_gram1}
            _ = sess.run(d_optim, feed_dict={images: batch_images, z: batch_z, vessel: batch_vessel})
            _ = sess.run(g_optim, feed_dict=feed_dict_g)
            _ = sess.run(g_optim, feed_dict=feed_dict_g)

            errD_fake = d_loss_fake.eval({z: batch_z, vessel: batch_vessel})
            errD_real = d_loss_real.eval({images: batch_images, vessel: batch_vessel})
            errG = g_loss.eval(feed_dict_g)

            count = count + 1


if __name__ == '__main__':
    # remove_all_file(SAVE_PATH)
    main()


