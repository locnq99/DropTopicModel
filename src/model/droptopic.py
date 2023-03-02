import torch
import copy
import torch.nn as nn 
from model.reparameter import ReparameterTrick
torch.manual_seed(666)

class DropTopic(nn.Module):
    def __init__(self, config, word_embedding) -> None:
        super(DropTopic, self).__init__()
        self.config = config
        self.reparameter = ReparameterTrick(config)

        self.word_embedding = torch.from_numpy(
            word_embedding).float().to(self.config.device)
        self.word_embedding.requires_grad = False

        self.pi_weight = self._init_weight_decode(config.init_topic_dim)
        
        self.prior_alpha = self._init_prior_bernoulli(config.init_topic_dim)
        self.alpha = self._init_posterior_bernoulli(config.init_topic_dim)
        self.sigmoid = nn.Sigmoid()

    def _init_weight_decode(self, n_topics):
        pi_weight = torch.Tensor(n_topics,
                                self.config.word_embedding_dim)
        pi_weight = pi_weight.to(self.config.device)
        pi_weight = nn.Parameter(pi_weight)
        nn.init.xavier_uniform_(pi_weight.data)
        return pi_weight
    
    def _init_posterior_bernoulli(self, n_topics):
        if self.config.model == 'bern':
            alpha_init = 0.0
            alpha = torch.tensor([alpha_init] * n_topics)
            # alpha = torch.FloatTensor(n_topics).uniform_(-2,2)
        else:
            alpha = torch.FloatTensor(n_topics).uniform_(-1,1)

        alpha = alpha.to(self.config.device)
        alpha = nn.Parameter(alpha)
        return alpha
    
    def _init_prior_bernoulli(self, n_topics):
        """
        init prior bernoulli distribution
        """
        if self.config.init_prior_alpha == 'fix':
            droprate_prior = -2.
            prior_alpha = torch.tensor(
                [droprate_prior] * n_topics
            )
        elif self.config.init_prior_alpha == 'random':
            prior_alpha = torch.FloatTensor(n_topics).uniform_(-2.,2.)
        
        prior_alpha = prior_alpha.to(self.config.device)
        prior_alpha = nn.Parameter(prior_alpha, requires_grad = False)
        # prior_alpha = nn.Parameter(prior_alpha, requires_grad = True)
        
        return prior_alpha
    
    def _get_weight_model(self):
        self.prev_pi = copy.deepcopy(self.pi_weight.data).detach()
        self.prev_posterior_alpha = copy.deepcopy(self.alpha.data).detach()
        self.prev_prior_alpha = copy.deepcopy(self.prior_alpha.data).detach()

    def _copy_weight(self, lst_topic_freeze, n_topics_scale, idx_copy):
        pi_scale = self._init_weight_decode(n_topics_scale)
        prev_pi = copy.deepcopy(self.pi_weight.data)
        
        # copy toan bo weight 
        if self.config.copy_all_weight: 
            lst_idx = list(range(len(prev_pi)))
            pi_scale.data[lst_idx,:] = prev_pi
        else:
            # chi copy weight tot
            pi_scale.data[lst_topic_freeze,:] = prev_pi[lst_topic_freeze,:]

        pi_scale.data[idx_copy,:] = self.prev_pi[idx_copy,:]
        
        self.pi_weight = nn.Parameter(pi_scale.detach())
        
    def _reset_model_weight(self, lst_topic_freeze, n_topics, idx_copy):
        self.alpha = self._init_posterior_bernoulli(n_topics)
        self.prior_alpha = self._init_prior_bernoulli(n_topics)
        self._copy_weight(lst_topic_freeze, n_topics, idx_copy)
        
    def forward(self):
        beta = torch.mm(self.pi_weight, self.word_embedding.T)
        alpha = self.sigmoid(self.alpha)
        posterior_droprate = self.reparameter(alpha = alpha)
        
        beta_hat = torch.mm(posterior_droprate.diag(), beta)        
        return beta, beta_hat
