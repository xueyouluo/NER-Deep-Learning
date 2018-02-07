import pickle

import tensorflow as tf

from model.config import Config
from model.model import NERModel
from utils.data_utils import convert_sentence, result_to_json, add_external_words
from utils.model_utils import read_tag_vocab, read_vocab
from utils.train_utils import get_config_proto

if __name__ == "__main__":
    checkpoint_dir = '/data/xueyou/ner/ner_lstm_dim256_no_external_words_0201/'
    config = pickle.load(open(checkpoint_dir + "config.pkl",'rb'))
    config.mode = "inference"
    if config.external_word_file:
        add_external_words(config.external_word_file)

    word2id, id2word = read_vocab(config.vocab_file)
    tag2id, id2tag = read_tag_vocab(config.tag_vocab_file)
    seg2id, id2seg = read_tag_vocab(config.segment_vocab_file)

    with tf.Session(config=get_config_proto(log_device_placement=False)) as sess:
        model = NERModel(sess,config)
        model.build()
        model.restore_model(checkpoint_dir)
        while True:
            line = input(">>")
            words,length,segments = convert_sentence(line.strip(), word2id, seg2id, lower=True)
            decode,_ = model.inference([words],[length],[segments])
            print(result_to_json(line,[id2tag[t] for t in decode[0]]))
