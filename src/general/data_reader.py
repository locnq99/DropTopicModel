import pickle
import numpy as np 
import torch
from torch.utils.data import Dataset

class DataReader(Dataset):
    def __init__(self,bow, pretrain_embedding):
        super(DataReader, self).__init__()
        self.bow = bow
        self.pretrain_embedding = pretrain_embedding
        self.n_data = bow.shape[0]
    def __len__(self):
        return self.n_data
    
    def __getitem__(self,idx):
        bow = self.bow[idx].toarray().squeeze()
        pretrain = self.pretrain_embedding[idx]
        return bow, pretrain
    
def read_data_test(path, num_test):
    def read_data(path):
        with open(path,'rb') as f:
            data = pickle.load(f)
        return data
    
    lst_part1_vector = []
    bows_part1 = []
    wordinds2 = []
    wordcnts2 = []
    for i in range(num_test):
        filename_part1_vector = '%s/data_test_%d_part_1_vector.pkl'%(path, i+1)
        filename_part1 = '%s/data_test_%d_part_1.pkl'%(path, i+1)
        filename_part2 = '%s/data_test_%d_part_2.pkl'%(path, i+1)

        part1_vector = read_data(filename_part1_vector)
        part1 = read_data(filename_part1)
        part2 = read_data(filename_part2)
        wordinds_2, wordcnts_2 = [], []
        for j in range(part2.shape[0]):
            dense_vector = part2[0].toarray()[0]
            inds = dense_vector.nonzero()[0]
            cnts = dense_vector[inds]
            
            wordinds_2.append(inds)
            wordcnts_2.append(cnts)

        lst_part1_vector.append(torch.from_numpy(part1_vector))
        bows_part1.append(torch.from_numpy(part1.toarray()))
        wordinds2.append(wordinds_2)
        wordcnts2.append(wordcnts_2)

    return lst_part1_vector, bows_part1, wordinds2, wordcnts2

def read_setting(path):
    with open(path + 'setting.txt','r') as f:
        setting = f.read().splitlines()
    config = {}
    for s in setting:
        tmp = s.split(':')
        if tmp[0] == 'batch_size':
            config['batch_size'] = int(tmp[1].strip())
        if tmp[0] == 'num_topic':
            config['num_topic'] = int(tmp[1].strip())
        if tmp[0] == 'num_test':
            config['num_test'] = int(tmp[1].strip())
        if tmp[0] == 'n_infer':
            config['n_infer_pp'] = int(tmp[1].strip())
        if tmp[0] == 'alpha':
            config['alpha_pp'] = float(tmp[1].strip())
            
    return config
    