import os, time, pickle, sys, math,random
import numpy as np
import hickle as hkl


data_path = sys.argv[1].rstrip('/')
save_dir = sys.argv[2].rstrip('/')
if not os.path.exists(data_path):
    print 'Data path wrong!'
    sys.exit()
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def hic_matrix_extraction(DPATH,res=10000,norm_method='NONE'):
    chrom_list = list(range(1,23))#chr1-chr22
    hr_contacts_dict={}
    for each in chrom_list:
        hr_hic_file = '%s/intra_%s/chr%d_10k_intra_%s.txt'%(DPATH,norm_method,each,norm_method)
        chrom_len = {item.split()[0]:int(item.strip().split()[1]) for item in open('chromosome.txt').readlines()}
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
        chrom_len = {item.split()[0]:int(item.strip().split()[1]) for item in open('chromosome.txt').readlines()}
        mat_dim = int(math.ceil(chrom_len['chr%d'%each]*1.0/res))
        lr_contact_matrix = np.zeros((mat_dim,mat_dim))
        for line in open(lr_hic_file).readlines():
            idx1, idx2, value = int(line.strip().split('\t')[0]),int(line.strip().split('\t')[1]),float(line.strip().split('\t')[2])
            lr_contact_matrix[idx1/res][idx2/res] = value
        lr_contact_matrix+= lr_contact_matrix.T - np.diag(lr_contact_matrix.diagonal())
        lr_contacts_dict['chr%d'%each] = lr_contact_matrix

    nb_hr_contacts={item:sum(sum(hr_contacts_dict[item])) for item in hr_contacts_dict.keys()}
    nb_lr_contacts={item:sum(sum(lr_contacts_dict[item])) for item in lr_contacts_dict.keys()}
    
    return hr_contacts_dict,lr_contacts_dict,nb_hr_contacts,nb_lr_contacts

hr_contacts_dict,lr_contacts_dict,nb_hr_contacts,nb_lr_contacts = hic_matrix_extraction(data_path)

max_hr_contact = max([nb_hr_contacts[item] for item in nb_hr_contacts.keys()])
max_lr_contact = max([nb_lr_contacts[item] for item in nb_lr_contacts.keys()])

hr_contacts_norm_dict = {item:np.log2(hr_contacts_dict[item]*max_hr_contact/sum(sum(hr_contacts_dict[item]))+1) for item in hr_contacts_dict.keys()}
lr_contacts_norm_dict = {item:np.log2(lr_contacts_dict[item]*max_lr_contact/sum(sum(lr_contacts_dict[item]))+1) for item in lr_contacts_dict.keys()}

max_hr_contact_norm={item:hr_contacts_norm_dict[item].max() for item in hr_contacts_dict.keys()}
max_lr_contact_norm={item:lr_contacts_norm_dict[item].max() for item in lr_contacts_dict.keys()}

hkl.dump(nb_hr_contacts,'%s/nb_hr_contacts.hkl'%save_dir)
hkl.dump(nb_lr_contacts,'%s/nb_lr_contacts.hkl'%save_dir)


hkl.dump(max_hr_contact_norm,'%s/max_hr_contact_norm.hkl'%save_dir)
hkl.dump(max_lr_contact_norm,'%s/max_lr_contact_norm.hkl'%save_dir)



def crop_hic_matrix_by_chrom(chrom,norm_type,size=40 ,thred=200):
    #thred=2M/resolution
    #norm_type=0-->raw count
    #norm_type=1-->log transformation
    #norm_type=2-->scaled to[-1,1]after log transformation
    #norm_type=3-->scaled to[0,1]after log transformation
    distance=[]
    crop_mats_hr=[]
    crop_mats_lr=[]    
    row,col = hr_contacts_norm_dict[chrom].shape
    if row<=thred or col<=thred:
        print 'HiC matrix size wrong!'
        sys.exit()
    def quality_control(mat,thred=0.05):
        if len(mat.nonzero()[0])<thred*mat.shape[0]*mat.shape[1]:
            return False
        else:
            return True
        
    for idx1 in range(0,row-size,size):
        for idx2 in range(0,col-size,size):
            if abs(idx1-idx2)<thred:
                if quality_control(lr_contacts_norm_dict[chrom][idx1:idx1+size,idx2:idx2+size]):
                    distance.append([idx1-idx2,chrom])
                    if norm_type==0:
                        lr_contact = lr_contacts_dict[chrom][idx1:idx1+size,idx2:idx2+size]
                        hr_contact = hr_contacts_dict[chrom][idx1:idx1+size,idx2:idx2+size]
                    elif norm_type==1:
                        lr_contact = lr_contacts_norm_dict[chrom][idx1:idx1+size,idx2:idx2+size]
                        hr_contact = hr_contacts_norm_dict[chrom][idx1:idx1+size,idx2:idx2+size]
                    elif norm_type==2:
                        lr_contact_norm = lr_contacts_norm_dict[chrom][idx1:idx1+size,idx2:idx2+size]
                        hr_contact_norm = hr_contacts_norm_dict[chrom][idx1:idx1+size,idx2:idx2+size]
                        lr_contact = lr_contact_norm*2.0/max_lr_contact_norm[chrom]-1
                        hr_contact = hr_contact_norm*2.0/max_hr_contact_norm[chrom]-1
                    elif norm_type==3:
                        lr_contact_norm = lr_contacts_norm_dict[chrom][idx1:idx1+size,idx2:idx2+size]
                        hr_contact_norm = hr_contacts_norm_dict[chrom][idx1:idx1+size,idx2:idx2+size]
                        lr_contact = lr_contact_norm*1.0/max_lr_contact_norm[chrom]
                        hr_contact = hr_contact_norm*1.0/max_hr_contact_norm[chrom]
                    else:
                        print 'Normalization wrong!'
                        sys.exit()
                    
                    crop_mats_lr.append(lr_contact)
                    crop_mats_hr.append(hr_contact)
    crop_mats_hr = np.concatenate([item[np.newaxis,:] for item in crop_mats_hr],axis=0)
    crop_mats_lr = np.concatenate([item[np.newaxis,:] for item in crop_mats_lr],axis=0)
    return crop_mats_hr,crop_mats_lr,distance
def data_split(chrom_list,norm_type):
    random.seed(100)
    distance_all=[]
    assert len(chrom_list)>0
    hr_mats,lr_mats=[],[]
    for chrom in chrom_list:
        crop_mats_hr,crop_mats_lr,distance = crop_hic_matrix_by_chrom(chrom,norm_type,size=40 ,thred=200)
        distance_all+=distance
        hr_mats.append(crop_mats_hr)
        lr_mats.append(crop_mats_lr)
    hr_mats = np.concatenate(hr_mats,axis=0)
    lr_mats = np.concatenate(lr_mats,axis=0)
    hr_mats=hr_mats[:,np.newaxis]
    lr_mats=lr_mats[:,np.newaxis]
    hr_mats=hr_mats.transpose((0,2,3,1))
    lr_mats=lr_mats.transpose((0,2,3,1))
    return hr_mats,lr_mats,distance_all


# hr_mats_train,lr_mats_train,distance_train = data_split(['chr%d'%idx for idx in list(range(1,18))],norm_type=0)
# hr_mats_test,lr_mats_test,distance_test = data_split(['chr%d'%idx for idx in list(range(18,23))],norm_type=0)
# hkl.dump([lr_mats_train,hr_mats_train,distance_train],'%s/train_data_raw_count.hkl'%save_dir)
# hkl.dump([lr_mats_test,hr_mats_test,distance_test],'%s/test_data_raw_count.hkl'%save_dir)

# hr_mats_train,lr_mats_train,distance_train = data_split(['chr%d'%idx for idx in list(range(1,18))],norm_type=1)
# hr_mats_test,lr_mats_test,distance_test = data_split(['chr%d'%idx for idx in list(range(18,23))],norm_type=1)
# hkl.dump([lr_mats_train,hr_mats_train,distance_train],'%s/train_data_log_trans.hkl'%save_dir)
# hkl.dump([lr_mats_test,hr_mats_test,distance_test],'%s/test_data_log_trans.hkl'%save_dir)

# hr_mats_train,lr_mats_train,distance_train = data_split(['chr%d'%idx for idx in list(range(1,18))],norm_type=2)
# hr_mats_test,lr_mats_test,distance_test = data_split(['chr%d'%idx for idx in list(range(18,23))],norm_type=2)
# hkl.dump([lr_mats_train,hr_mats_train,distance_train],'%s/train_data_log_trans_scaled_sym.hkl'%save_dir)
# hkl.dump([lr_mats_test,hr_mats_test,distance_test],'%s/test_data_log_trans_scaled_sym.hkl'%save_dir)

# hr_mats_train,lr_mats_train,distance_train = data_split(['chr%d'%idx for idx in list(range(1,18))],norm_type=3)
# hr_mats_test,lr_mats_test,distance_test = data_split(['chr%d'%idx for idx in list(range(18,23))],norm_type=3)
# hkl.dump([lr_mats_train,hr_mats_train,distance_train],'%s/train_data_log_trans_scaled_asym.hkl'%save_dir)
# hkl.dump([lr_mats_test,hr_mats_test,distance_test],'%s/test_data_log_trans_scaled_asym.hkl'%save_dir)


























































