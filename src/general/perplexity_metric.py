import numpy as np
from scipy.special import digamma
import warnings
warnings.filterwarnings('error')

def dirichlet_expectation(alpha):
    if(len(alpha.shape) == 1):
        return digamma(alpha) - digamma(sum(alpha))
    return (digamma(alpha) - digamma(np.sum(alpha, axis=1))[:, np.newaxis])

class Stream:
    def __init__(self, alpha, beta, num_topic, n_infer):
        self.alpha = alpha
        self.beta = beta
        self.num_topic = num_topic
        self.n_infer = n_infer
    
    def do_e_step(self, batch_size, wordinds, wordcnts):
        gamma = np.random.gamma(100., 1./100., (batch_size, self.num_topic))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)

        for d in range(batch_size):
            inds = wordinds[d]
            cnts = wordcnts[d]

            gamma_d = np.ones(self.num_topic) * self.alpha + float(np.sum(cnts)) / self.num_topic
            expElogtheta_d = np.exp(dirichlet_expectation(gamma_d))
            beta_d = self.beta[:, inds]
            for i in range(self.n_infer):
                phi_d = expElogtheta_d * beta_d.transpose() 
                phi_d /= np.sum(phi_d, axis=1)[:, np.newaxis]
                gamma_d = self.alpha + np.dot(cnts, phi_d)
                expElogtheta_d = np.exp(dirichlet_expectation(gamma_d))
                
            gamma[d] = gamma_d / sum(gamma_d) #normalization
    
        return gamma

    def compute_doc(self, gamma_d, wordinds, wordcnts):
        # gamma_d, wordinds, wordcnts = data
        ld2 = 0
        frequency = np.sum(wordcnts)
        for i in range(len(wordinds)):
            p = np.dot(gamma_d, self.beta[:, wordinds[i]])
            ld2 += wordcnts[i] * np.log(p)
        if(frequency == 0):
            return ld2
        else:
            return ld2/frequency
    
    def compute_perplexity(self, wordinds1, wordcnts1, wordinds2, wordcnts2):
        batchsize = len(wordinds1)
        LD2 = []
        gamma = self.do_e_step(batchsize, wordinds1, wordcnts1) 
        for i in range(batchsize):
            LD2.append(self.compute_doc(gamma[i], wordinds2[i], wordcnts2[i]))
        
        return sum(LD2)/batchsize