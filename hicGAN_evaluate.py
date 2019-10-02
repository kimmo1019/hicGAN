import os, time, pickle, random, time, sys, math
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
import hickle as hkl
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import matplotlib.pyplot as plt
from skimage.measure import compare_mse
from skimage.measure import compare_ssim

usage='''
Usage: python hicGAN_evaluate.py [GPU_ID] [MODEL_FOLDER] [CELL]
-- a program for evaluating hicGAN
[GPU_ID] : GPU ID 
[MODEL_FOLDER]: folder path containing weights file for hicGAN_g(e.g. checkpoint)
[CELL]: selected cell type (e.g. GM12878)
'''
if len(sys.argv)!=4:
    print usage
    sys.exit(1)

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
model_path=sys.argv[2].rstrip('/')
cell=sys.argv[3]


def calculate_psnr(mat1,mat2):
    data_range=np.max(mat1)-np.min(mat1)
    err=compare_mse(mat1,mat2)
    return 0 if err==0 else 10 * np.log10((data_range ** 2) / err)

def calculate_ssim(mat1,mat2):
    data_range=np.max(mat1)-np.min(mat1)
    return compare_ssim(mat1,mat2,data_range=data_range)

def hicGAN_g(t_image, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("hicGAN_g", reuse=reuse) as vs:
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = n
        # B residual blocks
        for i in range(5):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
            n = nn
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, name='add3')
        # B residual blacks end. output shape: (None,w,h,64)
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
        n = Conv2d(n, 1, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        return n
    
t_image = tf.placeholder('float32', [None, None, None, 1], name='image_input')
net_g = hicGAN_g(t_image, is_train=False, reuse=False)   

def hicGAN_predict(model_name,batch=64):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=model_name, network=net_g)
    out = np.zeros(lr_mats_test.shape)
    for i in range(out.shape[0]/batch):
        out[batch*i:batch*(i+1)] = sess.run(net_g.outputs, {t_image: lr_mats_test[batch*i:batch*(i+1)]})
    out[batch*(i+1):] = sess.run(net_g.outputs, {t_image: lr_mats_test[batch*(i+1):]})
    return out
#Comment the following line and constuct lr_mats_test,hr_mats_test by your own if you want to using custom data.
lr_mats_test,hr_mats_test,_=hkl.load('data/%s/test_data.hkl'%cell)

mse_list=[]
for i in range(100,500,5):
    pre = hicGAN_predict('%s/g_hicgan_%d.npz'%(model_path,i))
    mse_list.append(np.median(map(compare_mse,pre[:,:,:,0],hr_mats_test[:,:,:,0])))

best_model_idx = 100+5*(mse_list.index(min(mse_list)))
sr_mats_pre = hicGAN_predict('%s/g_hicgan_%d.npz'%(model_path,best_model_idx))
np.save('data/%s/hicGAN_predicted.npy'%cell,sr_mats_pre)
    
mse_hicGAN_norm=map(compare_mse,hr_mats_test[:,:,:,0],sr_mats_pre[:,:,:,0])
psnr_hicGAN_norm=map(calculate_psnr,hr_mats_test[:,:,:,0],sr_mats_pre[:,:,:,0])
#ssim_hicGAN_norm=map(calculate_ssim,hr_mats_test[:,:,:,0],sr_mats_pre[:,:,:,0])
print('The model with smallest MSE is g_hicgan_%d.npz'%best_model_idx)
print 'mse_hicGAN_norm:%.5f'%np.median(mse_hicGAN_norm)
print 'psnr_hicGAN_norm:%.5f'%np.median(psnr_hicGAN_norm)
#print 'ssim_hicGAN_norm:%.5f'%np.median(ssim_hicGAN_norm)

  
