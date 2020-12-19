
import tensorflow as tf
import numpy as np
import random
from matplotlib import image as Image
from matplotlib import pyplot as plt
import glob
from skimage import io,transform
import os
import scipy.io as scio



class DataLoader(object):

    def __init__(self, batch_size,input_size=np.array([224,224]),sysstr = "Windows" ):
        # reading data list

        self.input_size=input_size
        random.seed(20190222)
        
        self.path_to_image = 'img_data/OURS/image/'
        self.path_to_label = 'img_data/OURS/label/'
        self.batch_size = batch_size
        self.MEAN_VALUE = np.array([127.11, 77.02, 50.64])
        self.MEAN_VALUE = np.reshape(np.tile(self.MEAN_VALUE, (input_size[0], input_size[1])), [input_size[0], input_size[1], 3])

        # self.MEAN_VALUE = np.transpose(self.MEAN_VALUE, (2, 0, 1))



        self.list_img = os.listdir(self.path_to_image)
        self.list_img.sort(key=lambda x:int(x[:-4]))
        self.list_label = os.listdir(self.path_to_label)
        self.list_label.sort(key=lambda x:int(x[:-4]))

        self.size = len(self.list_img)
        self.num_batches = int(self.size / self.batch_size)
        self.cursor = 0
        self.sysstr =sysstr



    def get_batch(self):  # Returns
        if self.cursor + self.batch_size > self.size:
            self.cursor = 0
            np.random.shuffle(self.list_img)

        img = []
        label = []

        for idx in range(self.batch_size):
            curr_file_image = self.list_img[self.cursor][:-3] + 'jpg'
            curr_file_label = self.list_img[self.cursor][:-3] + 'txt'

            if (self.sysstr == "Windows"):
                full_img_path = self.path_to_image+curr_file_image
                full_label_path=self.path_to_label+curr_file_label
            else:
                
                full_img_path = 'img_data/OURS/image/' + curr_file_image
                full_label_path = 'img_data/OURS/label/' + curr_file_label
                
            self.cursor += 1

            label_gt=open(full_label_path,'r').readline()
            if label_gt =='1':
                label.append([1,0])
            else:
                label.append( [0,1])

            inputimage = io.imread(full_img_path)  # value:0~255, (192,256,3), BRG
            inputimage = transform.resize(inputimage, (self.input_size[0],self.input_size[1]))  ## (height, width)
            inputimage = inputimage.astype(np.dtype(np.float32))

            img.append(inputimage)

        return np.array(img),np.array(label)


class DataLoader_vessel(object):

    def __init__(self, batch_size, input_size=np.array([224, 224]), sysstr="Windows",path='OURS_138'):
        # reading data list

        self.input_size = input_size
        random.seed(20190307)
        self.path = path
        self.path_to_image = 'img_data/OURS_138/image/'
        self.path_to_vessel = 'img_data/OURS_138/vessel/'
        self.batch_size = batch_size
        self.MEAN_VALUE = np.array([127.11, 77.02, 50.64])
        self.MEAN_VALUE = np.reshape(np.tile(self.MEAN_VALUE, (input_size[0], input_size[1])),
                                     [input_size[0], input_size[1], 3])

        # self.MEAN_VALUE = np.transpose(self.MEAN_VALUE, (2, 0, 1))

        self.   list_img = os.listdir(self.path_to_image)
        self.list_img.sort(key=lambda x: int(x[:-4]))
        self.list_vessel = os.listdir(self.path_to_vessel)
        self.list_vessel.sort(key=lambda x: int(x[:-4]))

        self.size = len(self.list_img)
        self.num_batches = int(self.size / self.batch_size)
        self.cursor = 0
        self.sysstr = sysstr


    def get_batch(self):  # Returns
        if self.cursor + self.batch_size > self.size:
            self.cursor = 0
            np.random.shuffle(self.list_img)

        img = []
        vessel= []
        img_name=[]
        for idx in range(self.batch_size):
            curr_file_image = self.list_img[self.cursor][:-3] + 'jpg'
            curr_file_vessel = self.list_img[self.cursor][:-3] + 'jpg'

            if (self.sysstr == "Windows"):
                full_img_path = self.path_to_image + curr_file_image
                full_vessel_path = self.path_to_vessel + curr_file_vessel
            else:

                full_img_path = 'img_data/OURS_138/image/' + curr_file_image
                full_vessel_path = 'img_data/OURS_138/vessel/' + curr_file_vessel

            self.cursor += 1

            inputimage = io.imread(full_img_path)  # value:0~255, (192,256,3), BRG
            inputimage = transform.resize(inputimage, (self.input_size[0], self.input_size[1]))  ## (height, width)
            inputimage = inputimage.astype(np.dtype(np.float32))
            img.append(inputimage)

            inputvessel = io.imread(full_vessel_path)  # value:0~255, (192,256,3), BRG
            inputvessel = transform.resize(inputvessel, (self.input_size[0], self.input_size[1]))  ## (height, width)
            inputvessel = inputvessel.astype(np.dtype(np.float32))
            vessel.append(inputvessel)
            img_name.append(self.list_img[self.cursor-1][:-4])


        return np.array(img),np.array(vessel),img_name









