import torch 
import torch.nn as nn 
torch.manual_seed(666)

class ReparameterTrick(nn.Module):
    def __init__(self, config) -> None:
        super(ReparameterTrick, self).__init__()
        self.config = config

    def _sample_gumbel(self, shape, eps  = 1e-20):
        g = torch.Tensor(shape).uniform_(0,1)
        return -torch.log(-torch.log(g + eps)+eps).to(self.config.device)

    def _gumbel_softmax(self, logits):
        g1 = self._sample_gumbel(logits.size())
        g2 = self._sample_gumbel(logits.size())
        
        comp1 = ((torch.log(logits) + g1) / self.config.temperature).to(self.config.device)
        comp2 = ((torch.log(1 - logits) + g2) / self.config.temperature).to(self.config.device)
        
        combined = torch.cat((comp1.unsqueeze(1), comp2.unsqueeze(1)), dim = 1) 
        max_comp = torch.max(combined, dim = 1)[0]
        pi_t = torch.exp(comp1 - max_comp)/ (torch.exp(comp2-max_comp) + torch.exp(comp1 - max_comp))

        return pi_t.to(self.config.device).float()

    def _sample_gauss(self, mu = None, log_var = None):
        if log_var is None:
            mu = torch.exp(mu)
            eps = mu.data.new(mu.size()).normal_()
            return torch.ones_like(mu) + mu * eps
        else:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)

    def forward(self, mu = None, log_var = None, alpha = None):
        if alpha is not None:
            # reparameter for bernoulli distribution
            if self.training:
                return self._gumbel_softmax(alpha)
            else:
                return alpha
        else:
            ## reparameter for gauss distribution
            if self.training:
                return self._sample_gauss(mu, log_var)
            else:
                return mu
