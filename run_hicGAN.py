
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


# In[126]:


#GPU setting and Global parameters
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"
#checkpoint = "checkpoint"
checkpoint = sys.argv[1]
#log_dir = "log"
log_dir = sys.argv[2]
graph_dir = sys.argv[3]
save_dir_gan = "samples"
tl.global_flag['mode']='hicgan'
tl.files.exists_or_mkdir(checkpoint)
tl.files.exists_or_mkdir(save_dir_gan)
tl.files.exists_or_mkdir(log_dir)
batch_size = 128
lr_init = 1e-5
beta1 = 0.9
## initialize G
#n_epoch_init = 100
n_epoch_init = 1
n_epoch = 3000
lr_decay = 0.1
decay_every = int(n_epoch / 2)
ni = int(np.sqrt(batch_size))


# In[77]:


#Data preparation and preprocessing
def hic_matrix_extraction(DPATH,res=10000,norm_method='NONE'):
    chrom_list = list(range(1,23))#chr1-chr22
    hr_contacts_dict={}
    for each in chrom_list:
        hr_hic_file = '%s/intra_%s/chr%d_10k_intra_%s.txt'%(DPATH,norm_method,each,norm_method)
        chrom_len = {item.split()[0]:int(item.strip().split()[1]) for item in open('%s/chromosome.txt'%DPATH).readlines()}
        mat_dim = int(math.ceil(chrom_len['chr%d'%each]*1.0/res))
        hr_contact_matrix = np.zeros((mat_dim,mat_dim))
        for line in open(hr_hic_file).readlines():
            idx1, idx2, value = int(line.strip().split('\t')[0]),int(line.strip().split('\t')[1]),float(line.strip().split('\t')[2])
            hr_contact_matrix[idx1/res][idx2/res] = value
        hr_contact_matrix+= hr_contact_matrix.T - np.diag(hr_contact_matrix.diagonal())
        hr_contacts_dict['chr%d'%each] = hr_contact_matrix
    lr_contacts_dict={}
    for each in chrom_list:
        lr_hic_file = '%s/intra_%s/chr%d_10k_intra_%s_downsample_ratio16.txt'%(DPATH,norm_method,each,norm_method)
        chrom_len = {item.split()[0]:int(item.strip().split()[1]) for item in open('%s/chromosome.txt'%DPATH).readlines()}
        mat_dim = int(math.ceil(chrom_len['chr%d'%each]*1.0/res))
        lr_contact_matrix = np.zeros((mat_dim,mat_dim))
        for line in open(lr_hic_file).readlines():
            idx1, idx2, value = int(line.strip().split('\t')[0]),int(line.strip().split('\t')[1]),float(line.strip().split('\t')[2])
            lr_contact_matrix[idx1/res][idx2/res] = value
        lr_contact_matrix+= lr_contact_matrix.T - np.diag(lr_contact_matrix.diagonal())
        lr_contacts_dict['chr%d'%each] = lr_contact_matrix

    nb_hr_contacts={item:sum(sum(hr_contacts_dict[item])) for item in hr_contacts_dict.keys()}
    nb_lr_contacts={item:sum(sum(lr_contacts_dict[item])) for item in lr_contacts_dict.keys()}
    max_hr_contact = max([nb_hr_contacts[item] for item in nb_hr_contacts.keys()])
    max_lr_contact = max([nb_lr_contacts[item] for item in nb_lr_contacts.keys()])
    
    return hr_contacts_dict,lr_contacts_dict,max_hr_contact,max_lr_contact

# In[78]:

#uncommnet if not loading data
# hr_contacts_dict,lr_contacts_dict,max_hr_contact,max_lr_contact = hic_matrix_extraction('/home/liuqiao/software/HiCPlus/data/GM12878_primary/aligned_read_pairs')

# hr_contacts_norm_dict = {item:np.log2(hr_contacts_dict[item]*max_hr_contact/sum(sum(hr_contacts_dict[item]))+1) for item in hr_contacts_dict.keys()}
# lr_contacts_norm_dict = {item:np.log2(lr_contacts_dict[item]*max_lr_contact/sum(sum(lr_contacts_dict[item]))+1) for item in lr_contacts_dict.keys()}
# # In[118]:


#Data preparation and preprocessing
def crop_hic_matrix_by_chrom(chrom, size=40 ,thred=200):
    #thred=2M/resolution
    crop_mats_hr=[]
    crop_mats_lr=[]
    row,col = hr_contacts_norm_dict[chrom].shape
    if row<=thred or col<=thred:
        print 'HiC matrix size wrong!'
        sys.exit()
    def quality_control(mat,thred=0.1):
        if len(mat.nonzero()[0])<thred*mat.shape[0]*mat.shape[1]:
            return False
        else:
            return True
        
    for idx1 in range(0,row-size,size):
        for idx2 in range(0,col-size,size):
            if abs(idx1-idx2)<thred:
                if quality_control(hr_contacts_norm_dict[chrom][idx1:idx1+size,idx2:idx2+size]):
                    crop_mats_lr.append(lr_contacts_norm_dict[chrom][idx1:idx1+size,idx2:idx2+size])
                    crop_mats_hr.append(hr_contacts_norm_dict[chrom][idx1:idx1+size,idx2:idx2+size])
    crop_mats_hr = np.concatenate([item[np.newaxis,:] for item in crop_mats_hr],axis=0)
    crop_mats_lr = np.concatenate([item[np.newaxis,:] for item in crop_mats_lr],axis=0)
    return crop_mats_hr,crop_mats_lr 
def training_data_split(train_chrom_list):
    random.seed(100)
    assert len(train_chrom_list)>0
    hr_mats_train,lr_mats_train=[],[]
    for chrom in train_chrom_list:
        crop_mats_hr,crop_mats_lr = crop_hic_matrix_by_chrom(chrom, size=40 ,thred=200)
        hr_mats_train.append(crop_mats_hr)
        lr_mats_train.append(crop_mats_lr)
    hr_mats_train = np.concatenate(hr_mats_train,axis=0)
    lr_mats_train = np.concatenate(lr_mats_train,axis=0)
    hr_mats_train=hr_mats_train[:,np.newaxis]
    lr_mats_train=lr_mats_train[:,np.newaxis]
    hr_mats_train=hr_mats_train.transpose((0,2,3,1))
    lr_mats_train=lr_mats_train.transpose((0,2,3,1))
    train_shuffle_list = list(range(len(hr_mats_train)))
    hr_mats_train = hr_mats_train[train_shuffle_list]
    lr_mats_train = lr_mats_train[train_shuffle_list]
    return hr_mats_train,lr_mats_train



# In[119]:


#hr_mats_train,lr_mats_train = training_data_split(['chr%d'%idx for idx in list(range(1,18))])
lr_mats_train,hr_mats_train = hkl.load('./train_data.hkl')

#hr_mats_train_scaled = [item*2.0/item.max()-1 for item in hr_mats_train]
#lr_mats_train_scaled = [item*2.0/item.max()-1 for item in lr_mats_train]



# In[123]:


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
        for i in range(3):
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
        #n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')

        #n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
        #n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')

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

        n = Conv2d(n, 64, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n64s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s2/b')
        #output shape: (None,w/2,h/2,64)
        n = Conv2d(n, 128, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s1/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n128s1/b')

        n = Conv2d(n, 128, (3, 3), (4, 4), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n128s2/c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n128s2/b')
        #output shape: (None,w/4,h/4,64)
        #n = Conv2d(n, 256, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s1/c')
        #n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n256s1/b')

        #n = Conv2d(n, 256, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s2/c')
        #n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n256s2/b')
        #output shape: (None,w/8,h/8,256)
        #n = Conv2d(n, 512, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n512s1/c')
        #n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n512s1/b')

        #n = Conv2d(n, 512, (3, 3), (2, 2), act=lrelu, padding='SAME', W_init=w_init, b_init=b_init, name='n512s2/c')
        #n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n512s2/b')
        #output shape: (None,w/16,h/16,512)
        n = FlattenLayer(n, name='f')
        n = DenseLayer(n, n_units=1024, act=lrelu, name='d1024')
        n = DenseLayer(n, n_units=1, name='out')

        logits = n.outputs
        n.outputs = tf.nn.sigmoid(n.outputs)

        return n, logits


# In[124]:


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


# In[128]:


###========================= train GAN (hicGAN) =========================###

f_out = open('%s/train.log'%log_dir,'w')
#f_out1 = open('%s/train_detail.log'%log_dir,'w')
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

    ## save model
    if (epoch <=5) or ((epoch != 0) and (epoch % 5 == 0)):
        tl.files.save_npz(net_g.all_params, name=checkpoint + '/g_{}_{}.npz'.format(tl.global_flag['mode'],epoch), sess=sess)
        tl.files.save_npz(net_d.all_params, name=checkpoint + '/d_{}_{}.npz'.format(tl.global_flag['mode'],epoch), sess=sess)


# In[131]:


#out = sess.run(net_g.outputs, {t_image: test_sample})






