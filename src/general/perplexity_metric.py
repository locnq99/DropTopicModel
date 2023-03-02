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

    def compute_doc(self, theta_d, wordinds, wordcnts):
        ld2 = 0
        frequency = np.sum(wordcnts)
        for i in range(len(wordinds)):
            p = np.dot(theta_d, self.beta[:, wordinds[i]])
            ld2 += wordcnts[i] * np.log(p)
        if(frequency == 0):
            return ld2
        else:
            return ld2/frequency
    
    def compute_perplexity(self, theta, wordinds2, wordcnts2):
        batchsize = len(wordinds2)
        LD2 = []

        for i in range(batchsize):
            LD2.append(self.compute_doc(theta[i], wordinds2[i], wordcnts2[i]))
        
        return sum(LD2)/batchsize