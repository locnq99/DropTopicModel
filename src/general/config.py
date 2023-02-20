import torch 

class Config:
    def __init__(self,
                init_topic_dim = 100,
                max_num_topic = 100,
                pretrain_encode_dim = None,
                active_func = 'relu',
                word_embedding_dim = 200,
                l2_weight = None,
                model = 'bern',
                vocab_size = None,
                temperature = 0.1,
                threshold_npmi = None,
                init_prior_alpha = 'random',
                droprate_topic = None,
                dropout = 0.1,
                n_topic_scale = 100,
                update_all_weight = False,
                lr = 1e-4,
                batch_size = 512,
                n_loop_each_batch = 10,
                n_infer_pp = None, 
                alpha_pp = None,
                num_test_file = None,
                path_beta = None,
                path_kq = None,
                path_checkpoint = None,
                device = None
                ) -> None:
        self.init_topic_dim = init_topic_dim
        self.max_num_topic = max_num_topic
        self.pretrain_encode_dim = pretrain_encode_dim
        self.active_func = active_func
        self.word_embedding_dim = word_embedding_dim
        self.l2_weight = l2_weight
        self.model = model
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.threshold_npmi = threshold_npmi
        self.init_prior_alpha = init_prior_alpha
        self.droprate_topic = droprate_topic
        self.dropout = dropout
        self.n_topic_scale = n_topic_scale
        self.update_all_weight = update_all_weight
        self.lr = lr
        self.batch_size = batch_size
        self.n_loop_each_batch = n_loop_each_batch
        self.n_infer_pp = n_infer_pp
        self.alpha_pp = alpha_pp
        self.num_test_file = num_test_file
        self.path_beta = path_beta
        self.path_kq = path_kq
        self.path_checkpoint = path_checkpoint
        if device == None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available()
                                                else 'cpu')
        else:
            self.device = device