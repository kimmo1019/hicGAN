
# coding: utf-8
import os, time, pickle, random, time, sys, math
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import matplotlib.pyplot as plt
import hickle as hkl




#GPU setting and Global parameters
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#checkpoint = "checkpoint"
checkpoint = sys.argv[1]
#log_dir = "log"
log_dir = sys.argv[2]
graph_dir = sys.argv[3]
cell = sys.argv[4]
save_dir_gan = "samples"
tl.global_flag['mode']='hicgan'
tl.files.exists_or_mkdir(checkpoint)
tl.files.exists_or_mkdir(save_dir_gan)
tl.files.exists_or_mkdir(log_dir)
batch_size = 128
lr_init = 1e-4
beta1 = 0.9
## initialize G
n_epoch_init = 1
n_epoch = 500
lr_decay = 0.1
decay_every = int(n_epoch / 2)
ni = int(np.sqrt(batch_size))



#load data or construct with your own data. 
#shape(lr_mats_train) = (nb_train,40,40,1), shape(hr_mats_train) = (nb_train,40,40,1)
lr_mats_train,hr_mats_train = hkl.load('data/%s/train_data.hkl'%cell)


# Model implementation
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
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n128s1/1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
        n = Conv2d(n, 1, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        return n
    
    
def hicGAN_d(t_image, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope("hicGAN_d", reuse=reuse) as vs:
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='n64s1/c')

        n = Conv2d(n, 64, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s2/c1')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s2/b1')
        #output shape: (None,w/2,h/2,64)
        n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b2')

        n = Conv2d(n, 64, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s2/c3')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s2/b3')
        #output shape: (None,w/4,h/4,64)
        n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c4')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b4')

        n = Conv2d(n, 64, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s2/c5')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s2/b5')
        #output shape: (None,w/8,h/8,256)
        n = Conv2d(n, 128, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n128s1/b')

        #n = Conv2d(n, 512, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n512s2/c')
        #n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n512s2/b')
        #output shape: (None,w/16,h/16,512)
        n = FlattenLayer(n, name='f')
        n = DenseLayer(n, n_units=1024, act=lrelu, name='d1024')
        n = DenseLayer(n, n_units=1, name='out')

        logits = n.outputs
        n.outputs = tf.nn.sigmoid(n.outputs)

        return n, logits

t_image = tf.placeholder('float32', [batch_size, 40, 40, 1], name='input_to_generator')
t_target_image = tf.placeholder('float32', [batch_size, 40, 40, 1], name='t_target_hic_image')

net_g = hicGAN_g(t_image, is_train=True, reuse=False)
net_d, logits_real = hicGAN_d(t_target_image, is_train=True, reuse=False)
_, logits_fake = hicGAN_d(net_g.outputs, is_train=True, reuse=True)

net_g_test = hicGAN_g(t_image, is_train=False, reuse=True)
d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
d_loss = d_loss1 + d_loss2
g_gan_loss = 1e-1 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
#g_loss = mse_loss + g_gan_loss
g_loss = g_gan_loss
g_vars = tl.layers.get_variables_with_name('hicGAN_g', True, True)
d_vars = tl.layers.get_variables_with_name('hicGAN_d', True, True)

with tf.variable_scope('learning_rate'):
    lr_v = tf.Variable(lr_init, trainable=False)

g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

#summary variables
tf.summary.scalar("d_loss1", d_loss1)
tf.summary.scalar("d_loss2", d_loss2)
tf.summary.scalar("d_loss", d_loss)
tf.summary.scalar("mse_loss", mse_loss)
tf.summary.scalar("g_gan_loss", g_gan_loss)
tf.summary.scalar("g_combine_loss", 5e-2*g_gan_loss+mse_loss)
merged_summary = tf.summary.merge_all()

#Model pretraining G

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
tl.layers.initialize_global_variables(sess)

#record variables for tensorboard visualization
summary_writer=tf.summary.FileWriter('%s'%graph_dir,graph=tf.get_default_graph())

# sess.run(tf.assign(lr_v, lr_init))
# print(" ** fixed learning rate: %f (for init G)" % lr_init)
# f_out = open('%s/pre_train.log'%log_dir,'w')
# for epoch in range(0, n_epoch_init + 1):
#     epoch_time = time.time()
#     total_mse_loss, n_iter = 0, 0
#     for idx in range(0, len(hr_mats_train_scaled)-batch_size, batch_size):
#         step_time = time.time()
#         b_imgs_input = lr_mats_train_scaled[idx:idx + batch_size]
#         b_imgs_target = hr_mats_train_scaled[idx:idx + batch_size]
#         #b_imgs_384 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, is_random=True)
#         #b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
#         ## update G
#         errM, _ = sess.run([mse_loss, g_optim_init], {t_image: b_imgs_input, t_target_image: b_imgs_target})
#         print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
#         total_mse_loss += errM
#         n_iter += 1
#     log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f\n" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
#     print(log)
#     f_out.write(log)
# f_out.close()
#out = sess.run(net_g_test.outputs, {t_image: test_sample})
#print("[*] save images")
#tl.vis.save_images(out, [ni, ni], save_dir_gan + '/train_%d.png' % epoch)
## save model
#if (epoch != 0) and (epoch % 10 == 0):
#tl.files.save_npz(net_g.all_params, name=checkpoint + '/g_{}_init_{}.npz'.format(tl.global_flag['mode'],epoch), sess=sess)

###========================= train GAN (hicGAN) =========================###

f_out = open('%s/train.log'%log_dir,'w')
for epoch in range(0, n_epoch + 1):
    ## update learning rate
    if epoch != 0 and (epoch % decay_every == 0):
        #new_lr_decay = lr_decay**(epoch // decay_every)
        new_lr_decay=1
        sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
        log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
        print(log)
    elif epoch == 0:
        sess.run(tf.assign(lr_v, lr_init))
        log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
        print(log)

    epoch_time = time.time()
    total_d_loss, total_g_loss, n_iter = 0, 0, 0

    for idx in range(0, len(hr_mats_train)-batch_size, batch_size):
        step_time = time.time()
        b_imgs_input = lr_mats_train[idx:idx + batch_size]
        b_imgs_target = hr_mats_train[idx:idx + batch_size]
        ## update D
        errD, _ = sess.run([d_loss, d_optim], {t_image: b_imgs_input, t_target_image: b_imgs_target})
        ## update G
        errG, errM, errA, _ = sess.run([g_loss, mse_loss, g_gan_loss, g_optim], {t_image: b_imgs_input, t_target_image: b_imgs_target})
        print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f  adv: %.6f)" %
              (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errA))
        total_d_loss += errD
        total_g_loss += errG
        n_iter += 1

    log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f\n" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter,
                                                                            total_g_loss / n_iter)
    print(log)
    f_out.write(log)
    #record variables
    summary=sess.run(merged_summary,{t_image: b_imgs_input, t_target_image: b_imgs_target})
    summary_writer.add_summary(summary, epoch)
    
    

    ## quick evaluation on test sample
#     if (epoch != 0) and (epoch % 5 == 0):
#         out = sess.run(net_g_test.outputs, {t_image: test_sample})  #; print('gen sub-image:', out.shape, out.min(), out.max())
#         print("[*] save images")
#         tl.vis.save_images(out, [ni, ni], save_dir_gan + '/train_%d.png' % epoch)

    ## save model every 5 epochs
    if (epoch <=5) or ((epoch != 0) and (epoch % 5 == 0)):
        tl.files.save_npz(net_g.all_params, name=checkpoint + '/g_{}_{}.npz'.format(tl.global_flag['mode'],epoch), sess=sess)
        tl.files.save_npz(net_d.all_params, name=checkpoint + '/d_{}_{}.npz'.format(tl.global_flag['mode'],epoch), sess=sess)

#out = sess.run(net_g.outputs, {t_image: test_sample})






