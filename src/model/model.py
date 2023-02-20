import copy
import pickle
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

from model.encoder import ExternalEncoder
from model.droptopic import DropTopic
torch.manual_seed(666)

class DropTopicModel(nn.Module):
    def __init__(self, config, word_embedding) -> None:
        super(DropTopicModel, self).__init__()
        self.config = config

        self.encoder = ExternalEncoder(config)
        self.droptopic = DropTopic(config = config,
                                word_embedding = word_embedding)

        self.bn_recon = nn.BatchNorm1d(config.vocab_size, affine = False)

        self.curr_epoch = 0
        self.idx_topic_to_epoch = defaultdict(list)
        self.epoch_to_idx_topic = defaultdict(list)
        self.lst_idx_topic_for_metric = []
        self.lst_idx_topic_freeze = []
        self.lst_l2_weight = []
        self.lst_idx_topic_l2_weight = []
        self.lst_copy_weight = []
        self.prev_npmi_l2 = dict()

    def _store_best_topic(self, lst_topic_npmi):
        """
        save index of best topic
        """
        def update_l2_weight(idx, lst_l2_weight,lst_topic_npmi,
                                lst_idx_topic_l2_weight):
            npmi_score = lst_topic_npmi[idx]
            if npmi_score >= self.config.threshold_npmi[0]:
                lst_l2_weight.append(1e5)
                lst_idx_topic_l2_weight.append(idx)
            elif npmi_score >= self.config.threshold_npmi[1]: 
                lst_l2_weight.append(1e3)
                lst_idx_topic_l2_weight.append(idx)
            elif npmi_score >= self.config.threshold_npmi[2]:
                lst_l2_weight.append(1e1)
                lst_idx_topic_l2_weight.append(idx)
            return lst_l2_weight, lst_idx_topic_l2_weight

        self.lst_copy_weight = []
        for idx in self.lst_idx_topic_l2_weight:
            if lst_topic_npmi[idx] < self.prev_npmi_l2[idx]:
                self.lst_copy_weight.append(idx)
        self.prev_npmi_l2 = dict()

        droprate = copy.deepcopy(self.droptopic.alpha.data)
        if self.config.model == 'bern':
            droprate =  1 - torch.sigmoid(droprate)

        # get index of good topic for freeze  and l2 weight
        idx_small_droprate = (droprate <= self.config.droprate_topic)\
                            .nonzero().view(-1).detach().cpu().numpy()
        idx_freeze_update = [x for x in idx_small_droprate \
                                if lst_topic_npmi[x] > self.config.threshold_npmi[2]]
        self.lst_idx_topic_freeze += list(idx_freeze_update)
        self.lst_idx_topic_freeze = list(set(self.lst_idx_topic_freeze))
        
        lst_l2_weight = []
        lst_idx_topic_l2_weight = []
        for idx in self.lst_idx_topic_freeze:
            if lst_topic_npmi[idx] >= self.config.threshold_npmi[2]:
                self.prev_npmi_l2[idx] = lst_topic_npmi[idx]
                lst_l2_weight , lst_idx_topic_l2_weight \
                                        = update_l2_weight(idx, lst_l2_weight, 
                                            lst_topic_npmi, lst_idx_topic_l2_weight)

        self.lst_l2_weight = lst_l2_weight
        self.lst_idx_topic_l2_weight = lst_idx_topic_l2_weight

        ###
        for idx in idx_small_droprate:
            if lst_topic_npmi[idx] >= self.config.threshold_npmi[2]:
                self.idx_topic_to_epoch[idx].append(self.curr_epoch)
                self.epoch_to_idx_topic[self.curr_epoch].append(idx)
        
        # with open(f'/data/datn/stream/data/test_{self.curr_epoch}.pkl','wb') as f:
        #     pickle.dump(self.idx_topic_to_epoch, f, protocol = pickle.HIGHEST_PROTOCOL)
        self.curr_epoch += 1

        lst_idx = list(self.idx_topic_to_epoch.keys())
        lst_value = np.array(
            [len(x) for x in list(self.idx_topic_to_epoch.values())])
        top_idx = np.argsort(lst_value)[::-1]
        # lst_idx = np.array(lst_idx)[top_idx][:self.config.max_num_topic]
        # self.lst_idx_topic_for_metric = lst_idx
        # tqdm.write(f'num topic: {len(lst_topic_npmi)}')
        # tqdm.write(f'len topic metric: {len(self.lst_idx_topic_for_metric)}')

        lst_idx = np.array(lst_idx)[top_idx]
        lst_value = lst_value[top_idx]
        topic2len = dict(zip(lst_idx, lst_value))

        for epoch in self.epoch_to_idx_topic:
            lst_topic = self.epoch_to_idx_topic[epoch]
            lst_cnt = [topic2len[x] for x in lst_topic]
            idx = np.argsort(lst_cnt)[::-1]
            lst_topic = list(np.array(lst_topic)[idx])
            self.epoch_to_idx_topic[epoch] = lst_topic
        
        lst_idx = []
        check_break = False
        i = 0
        while True:
            if check_break or i >= self.config.max_num_topic:
                break

            for epoch in self.epoch_to_idx_topic:
                lst_topic = self.epoch_to_idx_topic[epoch]
                if len(lst_idx) == self.config.max_num_topic:
                    check_break = True
                    break

                if len(lst_topic) >= i + 1:
                    lst_idx.append(lst_topic[i])
                lst_idx = list(set(lst_idx))
            i += 1
                

        self.lst_idx_topic_for_metric = lst_idx
        # print(self.lst_idx_topic_for_metric)

        if len(self.lst_idx_topic_for_metric) < self.config.max_num_topic:
            tmp = set(list(range(len(lst_topic_npmi)))).difference(set(lst_idx))
            tmp = list(tmp)
            idx_tmp = np.argsort(-lst_topic_npmi[tmp])
            tmp = list(np.array(tmp)[idx_tmp])
            self.lst_idx_topic_for_metric = list(self.lst_idx_topic_for_metric) + tmp
            self.lst_idx_topic_for_metric = self.lst_idx_topic_for_metric[:self.config.max_num_topic]

        tqdm.write(f'total topic {len(self.droptopic.alpha)}')
        tqdm.write(f'num topic freeze {len(self.lst_idx_topic_freeze)}')
        tqdm.write(f'num topic metric: {len(self.lst_idx_topic_for_metric)}')

        
    def _save_previous_weight(self):
        self.encoder._get_weight_model()
        self.droptopic._get_weight_model()
        
    def _reset_model(self):
        """
        reset topic, droptopic weight
        """
        # save index best topic
        # self._save_previous_weight()
        # self._store_best_topic(lst_npmi)
        
        num_topic_scale = max(len(self.lst_idx_topic_freeze) + self.config.n_topic_scale, 
                               self.config.init_topic_dim)
        # num_topic_scale = max(len(self.droptopic.alpha) + self.config.n_topic_scale, 
        #                     self.config.init_topic_dim)
        
        # copy and reset weight
        self.encoder._reset_model_weight(self.lst_idx_topic_l2_weight, num_topic_scale, self.lst_copy_weight)
        self.droptopic._reset_model_weight(self.lst_idx_topic_l2_weight, num_topic_scale, self.lst_copy_weight)

    def _loss_KL_bern(self):
        """
        KL of 2 bernoulli distribution
        """
        prior_alpha = torch.sigmoid(self.droptopic.prior_alpha)
        posterior_alpha = torch.sigmoid(self.droptopic.alpha)
        KL = posterior_alpha*torch.log(posterior_alpha/prior_alpha) + \
                    (1-posterior_alpha)*torch.log((1-posterior_alpha)/(1-prior_alpha))
        return KL.mean()

    def _loss_KL_theta(self, posterior_mu, posterior_logvar):
        """
        KL of 2 gaussion distribution
        """
        posterior_var = torch.exp(posterior_logvar)
        var_division = torch.sum(posterior_var / self.encoder.prior_logvar, dim = 1)
        diff = self.encoder.prior_mu - posterior_mu
        diff_term = torch.sum(
            (diff * diff) / self.encoder.prior_logvar, dim = 1
        )
        logvar_det_division = self.encoder.prior_logvar.log().sum() - posterior_logvar.sum(dim = 1)
        KL = 0.5 * ( 
            var_division + diff_term - len(self.encoder.prior_mu) + logvar_det_division
        )
        return KL.mean()
    
    def _loss_recontruction(self, bow, x_recon):
        RL = -torch.sum(bow * torch.log(x_recon + 1e-10), dim = 1)
        return RL.mean()
    
    def _loss_weight_regularization(self):
        # get curr weight
        curr_pi = self.droptopic.pi_weight 
        curr_post_mu = self.encoder.mu_encode.weight
        curr_post_logvar = self.encoder.logvar_encode.weight
        curr_prior_mu = self.encoder.prior_mu 
        curr_prior_logvar = self.encoder.prior_logvar
        
        lst_l2_weight = torch.Tensor(self.lst_l2_weight)\
                                .view(-1, 1).to(self.config.device)
        
        term_pi = lst_l2_weight * \
                    (curr_pi[self.lst_idx_topic_l2_weight,:] - self.droptopic.prev_pi[self.lst_idx_topic_l2_weight,:]) ** 2
        term_post_mu = lst_l2_weight * \
                    (curr_post_mu[self.lst_idx_topic_l2_weight,:] - self.encoder.prev_posterior_mu[self.lst_idx_topic_l2_weight, :]) ** 2
        term_post_logvar = lst_l2_weight * \
                    (curr_post_logvar[self.lst_idx_topic_l2_weight, :] - self.encoder.prev_posterior_logvar[self.lst_idx_topic_l2_weight, :]) ** 2
        # term_prior_mu = lst_l2_weight * \
        #             (curr_prior_mu[self.lst_idx_topic_l2_weight] - self.prev_prior_mu[self.lst_idx_topic_l2_weight]) ** 2
        # term_prior_logvar = lst_l2_weight * \
        #             (curr_prior_logvar[self.lst_idx_topic_l2_weight] - self.prev_prior_logvar[self.lst_idx_topic_l2_weight]) ** 2
        
        loss = term_pi.sum() + term_post_mu.sum() + term_post_logvar.sum()
                # + term_prior_mu.sum() + term_prior_logvar.sum()
        return loss/len(self.lst_l2_weight)
        
    def _loss_func(self, posterior_mu, posterior_logvar,x_recon, \
                   bow, weight_regu = True, print_loss = False):
        RL = self._loss_recontruction(bow, x_recon)
        KL_theta = self._loss_KL_theta(posterior_mu, posterior_logvar)
        KL_drop = self._loss_KL_bern()
        loss = RL + KL_theta + KL_drop
        if weight_regu:
            loss_regu = self._loss_weight_regularization()
            loss_regu *= self.config.l2_weight

            if print_loss:
                tqdm.write(f'loss: {loss}, loss_regu: {loss_regu}')
            loss += loss_regu
        return loss

    def forward(self, bow, pretrain_embedding, weight_regu = True, print_loss = False):
        mu, logvar, theta = self.encoder(pretrain_embedding)
        beta, beta_hat = self.droptopic()

        # x_recon = F.softmax(
        #     torch.mm(theta, beta_hat), dim = 1
        # )

        bow = bow / bow.sum(axis = 1).unsqueeze(1)

        x_recon = F.softmax(
            self.bn_recon(torch.mm(theta, beta_hat)), dim = 1
        )
        loss = self._loss_func(posterior_mu = mu, 
                                posterior_logvar = logvar,
                                x_recon = x_recon,
                                bow = bow, 
                                weight_regu = weight_regu,
                                print_loss = print_loss)
        return loss
    
    def get_topic(self):
        beta, _= self.droptopic.forward()

        # beta_use = beta[self.lst_topic_freeze]
        beta_use = copy.deepcopy(beta[self.lst_idx_topic_for_metric].detach())
        # beta_use = beta

        beta = torch.softmax(beta, axis = 1)
        beta_use = torch.softmax(beta_use, axis = 1)
        return beta_use, beta