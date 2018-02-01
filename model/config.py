class Config(object):
    def __init__(self):
        self.checkpoint_dir = "/tmp/ner"
        self.mode = 'train'
        self.num_gpus = 1
        self.external_word_file = None

        # training 
        self.learning_rate = 0.001
        self.optimizer = 'adam'
        self.max_gradient_norm = 5.0
        self.keep_prob = 0.7
        self.num_train_steps = 200000
        self.learning_decay = False
        self.start_decay_step = None
        self.decay_times = 10
        self.decay_factor = 0.98
        self.batch_size = 32

        # embedding
        self.vocab_file = "vocab.txt"
        self.pretrained_embedding_file = None
        self.embedding_size = 256

        # segement embedding, works only size > 0
        self.segement_embedding_size = 20
        self.segment_onehot = False
        self.segment_tag_num = 4
        self.segment_vocab_file = None

        # LSTM
        self.encode_cell_type = 'lstm'
        self.num_units = 256
        self.num_bi_layers = 1

        # crf
        self.num_tags = 5
        self.tag_vocab_file = None

        # best f1
        self.best_dev_f1 = -100
        self.best_test_f1 = -100