# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.python.framework import ops
import os
import scipy.misc
import saliency
from alexnet import *
from matplotlib import image as image
from scipy.misc import imread, imresize
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from skimage import io,transform
from data_processing import DataLoader_vessel as DataLoader
from RimLoss_style_light import dataset
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
from matplotlib import pyplot as plt
from scipy.misc import imread, imresize
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def main():

        g = tf.Graph()
        sess = tf.Session(config=config)
        X = tf.placeholder(tf.float32, shape=(1, 227, 227, 3))
        Y = tf.placeholder(tf.int64, [1])  # 标签值:one-hot标签值
        batch_size = tf.placeholder(tf.float32, shape=(), name='batch_size')
        net = AlexNet(X, num_classes=2, is_training=False)
        with slim.arg_scope(AlexNet.alexnet_v2_arg_scope()):
            k = net.alexnet_v3()
            logits = k[11]
            norm_grads_0 = tf.gradients(logits[:, 1], k[0])[0]




        saver = tf.train.Saver()
        model_file = tf.train.latest_checkpoint('./ckpt3/')
        saver.restore(sess, model_file)

        dataloader = DataLoader(batch_size=1, input_size=np.array([227, 227]), sysstr='Linux', path=dataset)

        npy_350 = np.load('npy_350.npy')
        for i in range(dataloader.num_batches):

            image2, GTlabel2, img_name = dataloader.get_batch()

            cnn_output ,norm_grads= sess.run([k[7][0],norm_grads_0[0]], feed_dict={X: image2, Y: [1], batch_size: 1})  # shape=[6,6,256]

            "grad_cam______________________________________________________"
            weights = np.mean(np.abs(norm_grads), axis=(0, 1))  #
            cam = np.zeros(cnn_output.shape[0: 2], dtype=np.float32)  # shape=[7,7]
            for K, w in enumerate(weights):
                # if npy_350[i,0,K]:
                # if (w-np.min(weights))<0.05*(np.max(weights)-np.min(weights)):
                    cam += 1 * cnn_output[:, :, K]

            cam1 = np.maximum(cam, 0)  # relu
            cam2 = cam1 / np.max(cam1)
            cam3_255 = cam2 * 255
            cam3_uint8 = np.array(cam3_255, dtype=np.uint8)
            cam3_uint8_H = imresize(cam3_uint8, (227, 227))
            image2_uint8_H = imresize(image2[0], (227, 227))
            plt.imshow(image2_uint8_H)
            plt.show()
            plt.imshow(cam3_uint8_H,cmap=plt.get_cmap('viridis'))
            plt.show()
            # image.imsave('temp/layer1/' + img_name[0] + '.jpg', image2_uint8_H)
            # image.imsave('temp/layer1/' + img_name[0] + '_feature_all_8' + '.jpg',cam3_uint8_H,cmap=plt.get_cmap('viridis'))
            print(i)





if __name__ == '__main__':


    main()