import os
import pickle
import time

import tensorflow as tf

from model.config import Config
from model.model import NERModel
from utils.data_utils import (Batch, convert_dataset, create_vocab, read_data,
                              save_vocab, segment_vocab, update_tag_scheme)
from utils.evaluate import evaluate
from utils.train_utils import get_config_proto

if __name__ == "__main__":
    DATA_DIR = "./data"
    checkpoint_dir = '/tmp/ner/'

    # read training data
    train_data = read_data(os.path.join(DATA_DIR,"example.train"))

    # convert tags to iobes
    update_tag_scheme(train_data)

    # create vocab from training data
    word_vocab,tag_vocab = create_vocab(train_data)
    segment_vocab = segment_vocab()

    # save vocab
    save_vocab(word_vocab, os.path.join(checkpoint_dir,"word.vocab"))
    save_vocab(tag_vocab,os.path.join(checkpoint_dir,"tag.vocab"))
    save_vocab(segment_vocab,os.path.join(checkpoint_dir,"seg.vocab"))

    # convert word into ids
    train_data = convert_dataset(train_data, word_vocab, tag_vocab, segment_vocab)
    print("training data size: {0}".format(len(train_data)))

    # load test and dev data
    test_data = read_data(os.path.join(DATA_DIR,"example.test"))
    update_tag_scheme(test_data)
    test_data = convert_dataset(test_data, word_vocab, tag_vocab, segment_vocab)
    test_data_batch = Batch(test_data,200)

    dev_data = read_data(os.path.join(DATA_DIR,"example.dev"))
    update_tag_scheme(dev_data)
    dev_data = convert_dataset(dev_data, word_vocab, tag_vocab, segment_vocab)
    dev_data_batch = Batch(dev_data,200)

    # create model
    config = Config()

    # update the config
    if os.path.exists(os.path.join(checkpoint_dir + "config.pkl")):
        config = pickle.load(open(os.path.join(checkpoint_dir + "config.pkl"),'rb'))
    else:
        config.checkpoint_dir = checkpoint_dir
        config.vocab_file = os.path.join(checkpoint_dir,"word.vocab")
        config.pretrained_embedding_file = os.path.join(DATA_DIR,"wiki_100.utf8")
        config.num_tags = len(tag_vocab)
        config.segment_vocab_file = os.path.join(checkpoint_dir,"seg.vocab")
        config.tag_vocab_file = os.path.join(checkpoint_dir,"tag.vocab")
        pickle.dump(config, open(os.path.join(config.checkpoint_dir,"config.pkl"),'wb'))

    with tf.Session(config=get_config_proto()) as sess:
        model = NERModel(sess, config)
        model.build()
        model.init()

        try:
            model.restore_model()
            print("restored model from checkpoint dir")
        except:
            pass
        batch_manager = Batch(train_data,32)
        checkpoint_loss = 0.0
        step_time = 0.0
        stats_per_step = 100
        stats_per_eval = 500
        best_dev_f1 = -100
        best_test_f1 = -100

        for i in range(30):
            for batch in batch_manager.next_batch():
                start_time = time.time()
                loss, global_step = model.train_one_batch(*zip(*batch))
                step_time += (time.time() - start_time)
                checkpoint_loss += loss

                if global_step % stats_per_step == 0:
                    print("# global step - {0}, step time - {1}, loss - {2}".format(global_step,step_time/stats_per_step, checkpoint_loss/stats_per_step))
                    checkpoint_loss = 0.0
                    step_time = 0.0

            if global_step % stats_per_eval:
                dev_f1 = evaluate(model,"dev",dev_data,word_vocab,tag_vocab)
                test_f1 = evaluate(model,'test',test_data,word_vocab,tag_vocab)

                if dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
                    config.best_dev_f1 = best_dev_f1
                    print("New best dev f1 - {0}".format(best_dev_f1))
                    model.save_model(config.checkpoint_dir + "/best_dev")
                    pickle.dump(config, open(os.path.join(config.checkpoint_dir,"config.pkl"),'wb'))

                if test_f1 > best_test_f1:
                    best_test_f1 = test_f1
                    config.best_test_f1 = best_test_f1
                    print("New best test f1 - {0}".format(best_test_f1))
                    model.save_model(config.checkpoint_dir + "/best_test")
                    pickle.dump(config, open(os.path.join(config.checkpoint_dir,"config.pkl"),'wb'))

            model.save_model()
