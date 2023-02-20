import copy
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from model.reparameter import ReparameterTrick
torch.manual_seed(666)


class ExternalEncoder(nn.Module):
    def __init__(self, config) -> None:
        super(ExternalEncoder, self).__init__()
        self.config = config
        self.active = self._get_active()
        self.reparameter = ReparameterTrick(config)
        
        # init mu, log variance encoder
        self.mu_encode = self._init_weight_encode(config.init_topic_dim)
        self.logvar_encode = self._init_weight_encode(config.init_topic_dim)

        self.bn_mu = self._init_batchnorm(config.init_topic_dim)
        self.bn_logvar = self._init_batchnorm(config.init_topic_dim)
        
        self.prior_mu, self.prior_logvar = self._init_prior_gauss(config.init_topic_dim)
        self.dropout = nn.Dropout(config.dropout)

    def _init_weight_encode(self, n_topics):
        weight_encode = nn.Linear(self.config.pretrain_encode_dim,
                                n_topics)
        nn.init.xavier_uniform_(weight_encode.weight)
        return weight_encode.to(self.config.device)
    
    def _init_batchnorm(self, n_topics):
        bn = nn.BatchNorm1d(n_topics, affine = False).to(self.config.device)
        return bn 

    def _init_prior_gauss(self, n_topics):
        """
        init prior gauss distribution
        """
        topic_prior_mean = 0.0
        prior_mu = torch.tensor(
            [topic_prior_mean] * n_topics
        )
        prior_mu = prior_mu.to(self.config.device)
        prior_mu = nn.Parameter(prior_mu, requires_grad = False)

        topic_prior_var = 1. - (1./ n_topics)
        prior_var = torch.tensor(
            [topic_prior_var] * n_topics)
        prior_var = prior_var.to(self.config.device)
        prior_var = nn.Parameter(prior_var, requires_grad = False)
        return prior_mu, prior_var
    
    def _get_active(self):
        if self.config.active_func == 'softplus':
            return nn.Softplus()
        elif self.config.active_func == 'relu':
            return nn.ReLU()
    
    def _get_weight_model(self):
        self.prev_posterior_mu = copy.deepcopy(self.mu_encode.weight.data).detach()
        self.prev_posterior_logvar = copy.deepcopy(self.logvar_encode.weight.data).detach()
        self.prev_prior_mu = copy.deepcopy(self.prior_mu.data).detach()
        self.prev_prior_logvar = copy.deepcopy(self.prior_logvar.data).detach()
    
    def _reset_model_weight(self, lst_idx_copy_weight, n_topics_scale, lst_topic_freeze):
        mu_scale = self._init_weight_encode(n_topics_scale)
        logvar_scale = self._init_weight_encode(n_topics_scale)
        prior_mu, prior_var = self._init_prior_gauss(n_topics_scale)

        prev_mu = copy.deepcopy(self.mu_encode.weight.data)
        prev_logvar = copy.deepcopy(self.logvar_encode.weight.data)
        # prev_prior_mu = copy.deepcopy(self.prior_mu.data)
        # prev_prior_logvar = copy.deepcopy(self.prior_logvar.data)

        # copy toan bo weight
        if self.config.update_all_weight:
            lst_idx = list(range(len(prev_mu)))
            mu_scale.weight.data[lst_idx,:] = prev_mu
            logvar_scale.weight.data[lst_idx, :] = prev_logvar
             # prior_mu.data[lst_idx] = prev_prior_mu
            # prior_var.data[lst_idx] = prev_prior_logvar
        # copy 1 phan weight tot
        else: 
            mu_scale.weight.data[lst_idx_copy_weight, :] = prev_mu[lst_idx_copy_weight, :]
            logvar_scale.weight.data[lst_idx_copy_weight, :] = prev_logvar[lst_idx_copy_weight, :]

        mu_scale.weight.data[lst_topic_freeze,:] = self.prev_posterior_mu[lst_topic_freeze,:]
        logvar_scale.weight.data[lst_topic_freeze, :] = self.prev_posterior_logvar[lst_topic_freeze,:]
        
        #update
        # self.mu_encode = nn.Parameter(mu_scale.detach())
        # self.logvar_encode = nn.Parameter(logvar_scale.detach())
        self.mu_encode = mu_scale
        self.logvar_encode = logvar_scale
        self.prior_mu = nn.Parameter(prior_mu.detach())
        self.prior_logvar = nn.Parameter(prior_var.detach())

        self.bn_mu = self._init_batchnorm(n_topics_scale)
        self.bn_logvar = self._init_batchnorm(n_topics_scale)

    def forward(self, pretrain_embedding):
        mu = self.mu_encode(pretrain_embedding)
        mu = self.dropout(mu)
        mu = self.active(mu)

        logvar = self.logvar_encode(pretrain_embedding)
        logvar = self.dropout(logvar)
        logvar = self.active(logvar)

        mu = self.bn_mu(mu)
        logvar = self.bn_logvar(logvar)
        
        theta = F.softmax(
            self.reparameter(mu = mu, 
                            log_var = logvar), dim = 1
        )
        return mu, logvar, theta