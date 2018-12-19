import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import hickle as hkl
import os
import math
import gzip
import lasagne
from lasagne import layers
from nolearn.lasagne import NeuralNet
from lasagne.updates import sgd

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

DPATH=sys.argv[2]
#DPATH = '/home/liuqiao/homework/GAN/hicGAN/GM12878_primary'

X,Y,_ = hkl.load('%s/train_data_raw_count.hkl'%DPATH)
X =X.transpose((0,3,1,2))
Y_crop = Y[:,6:34,6:34,0]#40*40->28*28
Y_crop = Y_crop.reshape((Y_crop.shape[0], -1))
X = np.asarray(X,'float32')
Y_crop = np.asarray(Y_crop,'float32')


sample_size=X.shape[-1]
sys.setrecursionlimit(10000)
conv2d1_filters_numbers = 8
conv2d1_filters_size = 9
conv2d2_filters_numbers = 8
conv2d2_filters_size = 1
conv2d3_filters_numbers = 1
conv2d3_filters_size = 5

down_sample_ratio = 16
learning_rate = 0.00001
epochs = 10
HiC_max_value = 100
net1 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv2d1', layers.Conv2DLayer),
        ('conv2d2', layers.Conv2DLayer),
        ('conv2d3', layers.Conv2DLayer),
        ('output_layer', layers.FlattenLayer),
        ],
    input_shape=(None, 1, sample_size, sample_size),
    conv2d1_num_filters=conv2d1_filters_numbers, 
    conv2d1_filter_size = (conv2d1_filters_size, conv2d1_filters_size),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),  
    conv2d2_num_filters=conv2d2_filters_numbers, 
    conv2d2_filter_size = (conv2d2_filters_size, conv2d2_filters_size), 
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,   
    conv2d3_num_filters=conv2d3_filters_numbers, 
    conv2d3_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d3_filter_size = (conv2d3_filters_size, conv2d3_filters_size),
    update=sgd,       
    update_learning_rate = learning_rate,
    regression=True,
    max_epochs= epochs,
    verbose=1,
    )

net1.fit(X, Y_crop)

X_test,Y_test,distance_all= hkl.load('%s/test_data_raw_count.hkl'%DPATH)
X_test =X_test.transpose((0,3,1,2))
Y_test_crop = Y_test[:,6:34,6:34,0]
Y_test_crop = Y_test_crop.reshape((Y_test_crop.shape[0], -1))
X_test = np.asarray(X_test,'float32')
Y_test_crop = np.asarray(Y_test_crop,'float32')
Y_pre = net1.predict(X_test)
Y_pre = Y_pre.reshape((Y_pre.shape[0],28,28))
Y_test_crop = Y_test_crop.reshape((Y_pre.shape[0],28,28))
hkl.dump([Y_test_crop,Y_pre],'%s/hicPlus_pre.hkl'%DPATH)




















