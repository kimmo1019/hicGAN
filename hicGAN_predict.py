import os, time, pickle, random, time, sys, math
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
import hickle as hkl
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *


usage='''
Usage: python hicGAN_predict.py [GPU_ID] [MODEL_FOLDER]  [DATA_PATH] [SAVE_DIR]
-- a program for  hicGAN prediction
[GPU_ID] : GPU ID (e.g. 0)
[MODEL_PATH]:  path for weights file for hicGAN_g(e.g. checkpoint/g_hicgan_best.npz)
[DATA_PATH]: data path for enhancing (e.g. lr_mats_test.npy)
[SAVE_DIR]: save directory for predicted data 
'''
if len(sys.argv)!=5:
    print usage
    sys.exit(1)

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
model_name=sys.argv[2]
#constuct lr_mats_test by your own if you want to using custom data.
lr_mats_test=np.load(sys.argv[3]) #with shape (N,m,m,1)
save_dir = sys.argv[4].rstrip('/')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)





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

def hicGAN_predict(lr_mats_test,model_name,batch=64):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=model_name, network=net_g)
    out = np.zeros(lr_mats_test.shape)
    if out.shape[0] <= batch:
        out = sess.run(net_g.outputs, {t_image: lr_mats_test})
        return out
    else:
        for i in range(out.shape[0] // batch):
            out[batch*i:batch*(i+1)] = sess.run(net_g.outputs, {t_image: lr_mats_test[batch*i:batch*(i+1)]})
        out[batch*(i+1):] = sess.run(net_g.outputs, {t_image: lr_mats_test[batch*(i+1):]})
        return out


sr_mats_pre = hicGAN_predict(lr_mats_test,model_name)
np.save('%s/sr_mats_pre.npy'%save_dir,sr_mats_pre)
