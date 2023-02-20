import numpy as np 
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
        with open(path,'r') as f:
            data = f.read().splitlines()
        lst_wordinds = []
        lst_wordcnts = []
        for i in range(len(data)):
            tmp = data[i].split()
            tmp = tmp[1:]
            wordinds = []
            wordcnts = []
            for j in range(len(tmp)):
                ids, cnt = tmp[j].split(':')
                wordinds.append(ids)
                wordcnts.append(cnt)
            lst_wordinds.append(
                np.array(wordinds, dtype = np.int64))
            lst_wordcnts.append(
                np.array(wordcnts, dtype = np.int64))
        return lst_wordinds, lst_wordcnts
    
    wordinds1 = []
    wordcnts1 = []
    wordinds2 = []
    wordcnts2 = []
    for i in range(num_test):
        filename_part1 = '%s/data_test_%d_part_1.txt'%(path, i+1)
        filename_part2 = '%s/data_test_%d_part_2.txt'%(path, i+1)

        (wordinds_1, wordcnts_1) = read_data(filename_part1)
        (wordinds_2, wordcnts_2) = read_data(filename_part2)

        wordinds1.append(wordinds_1)
        wordcnts1.append(wordcnts_1)
        wordinds2.append(wordinds_2)
        wordcnts2.append(wordcnts_2)

    return wordinds1, wordcnts1, wordinds2, wordcnts2

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
    