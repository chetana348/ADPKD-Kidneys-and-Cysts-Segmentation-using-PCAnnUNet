import os
import numpy as np
import keras
import tensorflow
from sklearn.model_selection import train_test_split
import nibabel as nib
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import Sequence
from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from skimage import measure
from skimage.transform import resize
from PCAnnUNet.metrics import dice_coef
from PCAnnUNet import Custom_PCAnnUNet
from PCAnnUNet.losses import Hausdorff_distance
from sklearn.model_selection import train_test_split
from PIL import Image
from PIL import ImageOps
import fnmatch
import nibabel as nib
import shutil

class Parameter(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=12, dim=(512,512), n_channels=1,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    #we have rounded the number of total options (list_IDs)/batch size 
    #to get an integer for the length

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #print("indexes are as follows:"+str(indexes))
        #index*batch_size:index+1*batch size - block off a section the size of batchsize

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        #print("list IDs are as follows:"+str(list_IDs_temp))
        
        
        

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        
        y = np.empty((self.batch_size, *self.dim))
        

        
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #X[i,] = np.load('data/' + ID + '.npy')
            
            #if all of the files are in the same location use general directory
            #im_f_name = 'data\\' + ID]
            #maintain filepath in ID:
            im_f_name = ID
            lbl_f_name = im_f_name.replace('M.npy', 'K.npy')
            
            im = np.load(im_f_name)
            #print(im.max, im.min)
            lbl = np.load(lbl_f_name)
           
            
            
            
            X[i, ...,0] = im[...]
            y[i, ...] = lbl[...]
            

        return X, to_categorical(y, num_classes=self.n_classes)