import numpy as np
import scipy.misc
from alexnet import *
from matplotlib import pyplot as plt
from skimage import io,transform
from scipy.misc import imread, imresize
from data_processing import DataLoader_vessel as DataLoader
save_size=[1, 1]

def save_images(images, size, path):
    """

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


def main():
    imread


if __name__ == '__main__':
    # remove_all_file(SAVE_PATH)
    main()
