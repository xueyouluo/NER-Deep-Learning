import pickle

import tensorflow as tf

from model.config import Config
from model.model import NERModel
from utils.data_utils import convert_sentence, result_to_json
from utils.model_utils import read_tag_vocab, read_vocab
from utils.train_utils import get_config_proto

if __name__ == "__main__":
    checkpoint_dir = "/tmp/ner/"
    config = pickle.load(open(checkpoint_dir + "config.pkl",'rb'))
    config.mode = "inference"
    word2id, id2word = read_vocab(checkpoint_dir + "word.vocab")
    tag2id, id2tag = read_tag_vocab(checkpoint_dir + "tag.vocab")
    seg2id, id2seg = read_tag_vocab(checkpoint_dir + "seg.vocab")

    with tf.Session(config=get_config_proto(log_device_placement=False)) as sess:
        model = NERModel(sess,config)
        model.build()
        model.restore_model(checkpoint_dir)
        while True:
            line = input(">>")
            words,length,segments = convert_sentence(line.strip(), word2id, seg2id)
            decode,_ = model.inference([words],[length],[segments])
            print(result_to_json(line,[id2tag[t] for t in decode[0]]))
