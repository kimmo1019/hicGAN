import os, time, pickle, sys, math,random
import numpy as np
import scipy.sparse as sp
import hickle as hkl


#stat total counts of each chromosome
def get_stats_of_hic_count(cell,norm_method='NONE'):
    nb_hr_contacts,nb_lr_contacts={},{}
    chrom_list = list(range(1,23))#chr1-chr22
    for each in chrom_list:
        hr_hic_file = 'data/%s/intra_%s/chr%d_10k_intra_%s.txt'%(cell,norm_method,each,norm_method)
        lr_hic_file = 'data/%s/intra_%s/chr%d_10k_intra_%s_downsample_ratio16.txt'%(cell,norm_method,each,norm_method)
        cmd_hr = ''' awk -F "\\t" '{sum += $3};END {print sum}' %s'''%hr_hic_file
        cmd_lr = ''' awk -F "\\t" '{sum += $3};END {print sum}' %s'''%lr_hic_file
        nb_hr_contacts['chr%d'%each] = int(os.popen(cmd_hr).readlines()[0].strip())
        nb_lr_contacts['chr%d'%each] = int(os.popen(cmd_lr).readlines()[0].strip())
    return nb_hr_contacts,nb_lr_contacts
    

def hic_matrix_extraction(chrom, cell,res=10000,norm_method='NONE'):
    max_hr_contact = max(nb_hr_contacts.values())
    max_lr_contact = max(nb_lr_contacts.values())
    hr_hic_file = 'data/%s/intra_%s/%s_10k_intra_%s.txt'%(cell,norm_method,chrom,norm_method)
    mat_dim = int(math.ceil(chrom_len[chrom]*1.0/res))
    count,row,col=[],[],[]
    for line in open(hr_hic_file).readlines():
        idx1, idx2, value = int(line.strip().split('\t')[0]),int(line.strip().split('\t')[1]),float(line.strip().split('\t')[2])
        value = np.log2(value*max_hr_contact/nb_hr_contacts[chrom]+1)
        if idx1==idx2:
            count.append(value)
            row.append(idx1 // res)
            col.append(idx2 // res)
        else:
            count.append(value)
            row.append(idx1 // res)
            col.append(idx2 // res)
            count.append(value)
            row.append(idx2 // res)
            col.append(idx1 // res)
    hr_contact_sp_matrix = sp.csr_matrix((count,(row,col)),shape=(mat_dim,mat_dim))
    lr_hic_file = 'data/%s/intra_%s/%s_10k_intra_%s_downsample_ratio16.txt'%(cell,norm_method,chrom,norm_method)
    mat_dim = int(math.ceil(chrom_len[chrom]*1.0/res))
    count,row,col=[],[],[]
    for line in open(lr_hic_file).readlines():
        idx1, idx2, value = int(line.strip().split('\t')[0]),int(line.strip().split('\t')[1]),float(line.strip().split('\t')[2])
        value = np.log2(value*max_lr_contact/nb_lr_contacts[chrom]+1)
        if idx1==idx2:
            count.append(value)
            row.append(idx1 // res)
            col.append(idx2 // res)
        else:
            count.append(value)
            row.append(idx1 // res)
            col.append(idx2 // res)
            count.append(value)
            row.append(idx2 // res)
            col.append(idx1 // res)
    lr_contact_sp_matrix = sp.csr_matrix((count,(row,col)),shape=(mat_dim,mat_dim))
    return hr_contact_sp_matrix,lr_contact_sp_matrix


def crop_hic_matrix_by_chrom(chrom, size=40 ,thred=200):
    #thred=2M/resolution
    #norm-->scaled to[-1,1]after log transformation, default
    distance=[] #record the location of a cropped mat in test data
    crop_mats_hr=[]
    crop_mats_lr=[]    
    hr_contact_sp_matrix,lr_contact_sp_matrix = hic_matrix_extraction(chrom,cell)
    row,col = hr_contact_sp_matrix.shape
    if row<=thred or col<=thred:
        print 'HiC matrix size wrong!'
        sys.exit()
    def quality_control(sp_mat,thred=0.05):
        if len(sp_mat.data)<thred*np.product(sp_mat.shape):
            return False
        else:
            return True
    for idx1 in range(0,row-size,size):
        for idx2 in range(0,col-size,size):
            if abs(idx1-idx2)<thred:
                if quality_control(lr_contact_sp_matrix[idx1:idx1+size,idx2:idx2+size]):
                    distance.append([idx1-idx2,chrom])
                    lr_contact_crop = lr_contact_sp_matrix[idx1:idx1+size,idx2:idx2+size]
                    hr_contact_crop = hr_contact_sp_matrix[idx1:idx1+size,idx2:idx2+size]
                    lr_contact = lr_contact_crop.toarray()*2.0/lr_contact_sp_matrix.max() - 1
                    hr_contact = hr_contact_crop.toarray()*2.0/hr_contact_sp_matrix.max() - 1
                    crop_mats_lr.append(lr_contact)
                    crop_mats_hr.append(hr_contact)
    return crop_mats_hr,crop_mats_lr,distance

def data_split(chrom_list):
    random.seed(100)
    distance_all=[]
    assert len(chrom_list)>0
    hr_mats,lr_mats=[],[]
    for chrom in chrom_list:
        crop_mats_hr,crop_mats_lr,distance = crop_hic_matrix_by_chrom(chrom, size=40 ,thred=200)
        distance_all += distance
        hr_mats += crop_mats_hr
        lr_mats += crop_mats_lr
    hr_mats = np.stack(hr_mats)
    lr_mats = np.stack(lr_mats)
    hr_mats=hr_mats[:,np.newaxis]
    lr_mats=lr_mats[:,np.newaxis]
    hr_mats=hr_mats.transpose((0,2,3,1))
    lr_mats=lr_mats.transpose((0,2,3,1))
    return hr_mats,lr_mats,distance_all

if __name__=="__main__":
    cell=sys.argv[1]
    if not os.path.exists('data/%s'%cell):
        print 'Data path wrong,please input the right data path.'
        sys.exit()
    chrom_len = {item.split()[0]:int(item.strip().split()[1]) for item in open('chromosome.txt').readlines()}
    nb_hr_contacts,nb_lr_contacts = get_stats_of_hic_count(cell)
    hr_mats_train,lr_mats_train,distance_train = data_split(['chr%d'%idx for idx in list(range(1,18))])
    hr_mats_test,lr_mats_test,distance_test = data_split(['chr%d'%idx for idx in list(range(18,23))])
    hkl.dump([lr_mats_train,hr_mats_train],'data/%s/train_data.hkl'%cell)
    hkl.dump([lr_mats_test,hr_mats_test,distance_test],'data/%s/test_data.hkl'%cell)
