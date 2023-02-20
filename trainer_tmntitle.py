import pandas as pd 
import copy
from tqdm import tqdm
import numpy as np 
import pickle
from collections import defaultdict

import torch
from torch import optim

from model.model import DropTopicModel
from general.config import Config
from general.data_reader import read_data_test, read_setting
from general.compute_metric import compute_npmi, compute_perplexity
torch.manual_seed(666)


class Trainer:
    def __init__(self, config, word_embedding, wordinv):
        self.config = config
        self.net = DropTopicModel(config, word_embedding)
        self.net = self.net.to(self.config.device)

        # init optimizer
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=config.lr,  betas=(0.99, 0.995))
        
        self.best_loss_train = float('inf')
        self.kq = pd.DataFrame(columns=['epoch_id','loss', 'LPP', 'npmi'])
        self.word_co_occurrence = defaultdict(list)
        self.curr_num_docs = 0
        self.wordinv = wordinv
        
    def fit(self, train_bow, train_embedding, data_test):
        """
        input:
            train: type DataSet
            data_test: 
                tuple (wordinds1, wordcnts2, wordinds2, wordcnts2)
        """                      
        n  = train_bow.shape[0]
        epoch_size = self.config.batch_size
        n_epoch = int(n // epoch_size)
        for i in tqdm(range(n_epoch), desc = 'Training'):
            self.net.train()

            bows = train_bow[i * epoch_size : (i+1) * epoch_size]
            embedding = train_embedding[i * epoch_size: (i+1) * epoch_size]
            
            # update word cooccurrence
            self._update_word_co_occurrence(bows)
            
            bows = torch.from_numpy(bows.toarray())\
                        .to(self.config.device).float()
            embedding = torch.from_numpy(embedding)\
                            .to(self.config.device).float()

            n_loop = self.config.n_loop_each_batch
            for j in range(n_loop):
                idx_shuffle = torch.randperm(len(bows))
                bows = bows[idx_shuffle]
                embedding = embedding[idx_shuffle]
                self.optimizer.zero_grad()

                if j < n_loop - 5: # update theta
                    self.net.encoder.train()
                    self.net.droptopic.eval()
                else: # update beta
                    self.net.encoder.eval()
                    self.net.droptopic.train()

                if i == 0:
                    loss = self.net(bows, embedding, False, False)
                else: 
                    if (j + 1) % 10 == 0:
                        loss = self.net(bows, embedding, True, True)
                    else:
                        loss = self.net(bows, embedding, True, False)
                loss.backward()
                self.optimizer.step()
            
            self.net.eval()
            beta_use, all_beta = self.net.get_topic()
            npmi_score, lst_npmi = self._get_topic_coherence(all_beta)

            self.net._store_best_topic(lst_npmi)
            if i == 0:
                self.net._save_previous_weight()

            LD = 0
            if i %10 == 0:
                LD = self._update_result(0,0, data_test, 0)

            if i == n_epoch - 1:
                LD = self._update_result(0,0, data_test, 0)
            
            # print(lkdjhafkjh)
            if i != n_epoch - 1:
                self.net._reset_model()

            beta_use, all_beta = self.net.get_topic()
            npmi_score, lst_npmi = self._get_topic_coherence(all_beta)
            self._save_model(i, lst_npmi)
            
            if i != 0:
                self.net._save_previous_weight()

            beta_use, all_beta = self.net.get_topic()
            npmi_score, lst_npmi = self._get_topic_coherence(beta_use)
            tqdm.write(f'npmi after scale: {npmi_score}')
            tqdm.write('----------------------------------------')
            self._reset_optimizer(lr = self.config.lr)
            
            n = len(self.kq)
            self.kq.loc[n] = [i, loss.item(), LD, npmi_score]
            self.kq.to_csv(self.config.path_kq)

            # self.net.eval()
            # beta_use, all_beta = self.net.get_topic()
            # npmi_score, lst_npmi = self._get_topic_coherence(beta_use)
            
            # tqdm.write(f'num topic: {len(self.net.lst_topic)},\
            #                 npmi score: {npmi_score}')
            # print(dlhafkjhauj)
    
    def _update_word_co_occurrence(self, bows):
        num_news_docs, n_words = bows.shape

        for i in range(num_news_docs):
            idx = bows[i].toarray()[0].nonzero()[0]
            for j in range(len(idx)):
                if j != 0:
                    self.word_co_occurrence[idx[j]].append(i + self.curr_num_docs)
            
        self.curr_num_docs += num_news_docs
            
    def _reset_optimizer(self, lr):
        self.best_loss_train = float('inf')
        self.optimizer = optim.Adam(
            self.net.parameters(), lr = lr,  betas=(0.99, 0.995))

    def _store_pi_weight(self):
        pi_weight = copy.deepcopy(self.net.pi.data)
        return pi_weight

    def _get_topic_coherence(self, beta):
        npmi_score, lst_npmi = compute_npmi(beta.detach().cpu().numpy(),
                                        #     self.word_co_occurrence,
                                        #    self.curr_num_docs,
                                           self.wordinv,
                                            26251,
                                           20)
        return npmi_score, lst_npmi
        
    def _update_result(self, batch_id, loss, data_test, word_inv):
        wordinds1, wordcnts, wordinds2, wordcnts2 = data_test
        beta, _ = self.net.get_topic()
        # _, beta= self.net.get_topic()
        beta = beta.detach().cpu().numpy()
        
        LD, ld2 = compute_perplexity(wordinds1, wordcnts, wordinds2, wordcnts2, \
                    len(beta), self.config.n_infer_pp, self.config.alpha_pp, \
                        beta, self.config.num_test_file)
        tqdm.write(f'######################################################### LPP: {LD}')
        return LD
        # print(kjdahfkjha)
#         npmi_score, lst_npmi = compute_npmi(beta, word_inv, top_k = 20)
#         diversity_score = compute_diversity(beta, 25)
        
        # n = len(self.kq)
        # self.kq.loc[n] = [batch_id, loss, len(beta), LD, npmi_score, diversity_score]

#         self._update_pi_weight(lst_npmi)
    

    def _save_model(self, batch_id, lst_npmi):
        """
        Save model.

        Args
            
        """
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'batch_id': batch_id,
            'topic_npmi': lst_npmi,
            'lst_topic_freeze': self.net.lst_idx_topic_freeze,
            'lst_topic_metric': self.net.lst_idx_topic_for_metric
            }, self.config.path_checkpoint + f'_{batch_id}.pt')
        # save Beta
        beta, all_beta = self.net.get_topic()
        beta = beta.detach().cpu().numpy()
        all_beta = all_beta.detach().cpu().numpy()
        with open(self.config.path_beta + '_{}.pkl'.format(batch_id),'wb') as f:
            pickle.dump(beta,f,protocol = pickle.HIGHEST_PROTOCOL)
        with open(self.config.path_beta + '_all_{}.pkl'.format(batch_id), 'wb') as f:
            pickle.dump(all_beta, f, protocol = pickle.HIGHEST_PROTOCOL)
        # self.kq.to_csv(self.config.path_kq, index = False)            
            
if __name__ == "__main__":
    print('Load data')
    path_data = '/data/datn/final_data/holdout_data/TMNtitle/'
    
    #read config
    setting = read_setting(path_data)
    with open(path_data + 'train.pkl','rb') as f:
        bows = pickle.load(f)   
    with open(path_data + 'docs_vector.pkl','rb') as f:
        pretrain_embedding = pickle.load(f)
    with open(path_data + 'prior.pkl', 'rb') as f:
        word_embedding = pickle.load(f)
    
    data_test = read_data_test(path = path_data,
                              num_test = setting['num_test'])
    # get wordinv
    word_inv = defaultdict(list)
    for i in range(bows.shape[0]):
        idx = bows[i].toarray()[0].nonzero()[0]
        for j in range(len(idx)):
            if j != 0:
                word_inv[idx[j]].append(i)

    with open('/data/datn/stream/data/test.pkl','wb') as f:
        pickle.dump(word_inv, f, protocol = pickle.HIGHEST_PROTOCOL)
    
    print('Loader data done!')
    print('Setup config')
    config = Config(init_topic_dim = 50,
                    max_num_topic = setting['num_topic'],
                    pretrain_encode_dim = 200,
                    active_func = 'softplus',
                    word_embedding_dim = 200,
                    model = 'bern',
                    l2_weight = 2e-2,
                    vocab_size = bows.shape[1],
                    temperature = 0.1,
                    threshold_npmi = [0.04,0.02,0.01],
                    init_prior_alpha = 'fix',
                    droprate_topic = 0.56,
                    dropout = 0.2,
                    n_topic_scale = 2,
                    update_all_weight = True,
                    lr = 1e-2, 
                    batch_size = setting['batch_size'],
                    n_loop_each_batch = 11,
                    n_infer_pp = setting['n_infer_pp'],
                    alpha_pp = setting['alpha_pp'],
                    num_test_file = setting['num_test'],
                    path_beta = '../data/TMNtitle/beta',
                    path_kq = '../data/TMNtitle/kq.csv',
                    path_checkpoint = '../data/TMNtitle/checkpoint',
                    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    print('config done!')
    model = Trainer(config, word_embedding, word_inv)
    print('Init Model done!')
    print('Training')
    model.fit(bows, pretrain_embedding, data_test)
    
    
   