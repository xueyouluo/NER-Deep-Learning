class Config(object):
    checkpoint_dir = "/tmp/ner"
    mode = 'train'
    num_gpus = 1

    # training 
    learning_rate = 0.001
    optimizer = 'adam'
    max_gradient_norm = 5.0
    keep_prob = 0.75

    # embedding
    vocab_file = "test/vocab.txt"
    pretrained_embedding_file = None
    embedding_size = 100

    # segement embedding, works only size > 0
    segement_embedding_size = 10
    segment_onehot = True
    segment_tag_num = 4
    segment_vocab_file = None

    # LSTM
    encode_cell_type = 'lstm'
    num_units = 100
    num_bi_layers = 1

    # crf
    num_tags = 5
    tag_vocab_file = None

    # best f1
    best_dev_f1 = -100
    best_test_f1 = -100