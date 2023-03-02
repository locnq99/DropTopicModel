from general.perplexity_metric import Stream
from general.npmi_metric import get_topic_coherence, get_topic_diversity

def compute_perplexity(lst_theta, wordinds2, wordcnts2,\
        num_topics, n_infer, alpha, beta, num_test, model):
    perplex = Stream(alpha, beta, num_topics, n_infer)
    LD = 0
    ld2 = []
    
    for i in range(num_test):
        ld = perplex.compute_perplexity(lst_theta[i],\
                                wordinds2[i], wordcnts2[i])
        LD += ld
        ld2.append(ld)
    LD /= num_test
    return LD, ld2

def compute_npmi(beta, word_inv, num_docs, top_k = 20):
    """
    input:
    """
    # compute npmi score
    npmi_score = get_topic_coherence(beta, word_inv, num_docs, top_k)
    return npmi_score

def compute_diversity(beta, top_k = 25):
    return get_topic_diversity(beta, top_k)