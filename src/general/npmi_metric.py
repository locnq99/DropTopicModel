import numpy as np
from tqdm import tqdm
import itertools as it


def get_topic_diversity(beta, topk = 25):
    num_topics = beta.shape[0]
    if num_topics == 0:
        return 0
    list_w = np.zeros((num_topics, topk))
    for k in range(num_topics):
        idx = beta[k,:].argsort()[-topk:][::-1]
        list_w[k,:] = idx
    n_unique = len(np.unique(list_w))
    TD = n_unique / (topk * num_topics)
    return TD

def get_document_frequency(inv_idx, idx_wi, idx_wj=None):
    if idx_wj is None:
        D_wi = len(inv_idx[idx_wi])
        return D_wi
    else:
        lst_docs_wi = inv_idx[idx_wi]
        lst_docs_wj = inv_idx[idx_wj]
        D_wj = len(lst_docs_wj)
        D_wi_wj = len(set(lst_docs_wi).intersection(lst_docs_wj))
        return D_wj, D_wi_wj

def get_topic_coherence(beta, inv_idx, num_docs, top_k = 10):
    # D = len(data) ## number of docs...data is list of documents
    D = num_docs
    
    idx_top_word = np.argsort(-beta,axis = 1)
    idx_top_word = idx_top_word[:,:top_k]
    num_topics = len(beta)
    num_couple = top_k * (top_k - 1)/2

    lst_npmi = []
    eps = 0.01
    for i in range(num_topics):
        npmi = 0
        for couple in list(it.combinations(idx_top_word[i],2)):
            w_i = couple[0]
            w_j = couple[1]
            D_wi = get_document_frequency(inv_idx, w_i)
            D_wj, D_wi_wj = get_document_frequency(inv_idx, w_i, w_j)
            if D_wi == 0:
                D_wi = eps
            if D_wj == 0:
                D_wj = eps
            if D_wi_wj == 0:
                npmi_couple = np.log(1.0 * D_wi * D_wj / num_docs**2) / np.log(1.0 * eps / num_docs) - 1
            else:
                npmi_couple = np.log(1.0 * D_wi * D_wj / num_docs**2) / np.log(1.0 * D_wi_wj / num_docs) - 1
            npmi += npmi_couple
        lst_npmi.append(npmi/num_couple)

    return np.mean(lst_npmi), np.array(lst_npmi)