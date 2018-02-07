import os
import pickle
import time

import tensorflow as tf

from model.config import Config
from model.model import NERModel
from utils.data_utils import (Batch, convert_dataset, create_vocab, read_data,
                              save_vocab, segment_vocab, update_tag_scheme, add_external_words)
from utils.evaluate import evaluate
from utils.train_utils import get_config_proto

if __name__ == "__main__":
    DATA_DIR = "/data/xueyou/fashion/data/"
    checkpoint_dir = '/data/xueyou/ner/category_ner_lstm_dim256_0207/'

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # read training data
    train_files = [os.path.join(DATA_DIR,"category.ner.train.txt")]
    train_data = read_data(train_files,lower=True)

    # convert tags to iobes
    update_tag_scheme(train_data)

    # create vocab from training data
    word_vocab,tag_vocab = create_vocab(train_data, lower_case=True)
    segment_vocab = segment_vocab()

    # save vocab
    save_vocab(word_vocab, os.path.join(checkpoint_dir,"word.vocab"))
    save_vocab(tag_vocab,os.path.join(checkpoint_dir,"tag.vocab"))
    save_vocab(segment_vocab,os.path.join(checkpoint_dir,"seg.vocab"))

    # convert word into ids
    train_data = convert_dataset(train_data, word_vocab, tag_vocab, segment_vocab)
    print("training data size: {0}".format(len(train_data)))

    # load test and dev data
    test_data = read_data(os.path.join(DATA_DIR,"category.ner.test.txt"),lower=True)
    update_tag_scheme(test_data)
    test_data = convert_dataset(test_data, word_vocab, tag_vocab, segment_vocab)
    test_data_batch = Batch(test_data,500)

    dev_data = read_data(os.path.join(DATA_DIR,"category.ner.dev.txt"),lower=True)
    update_tag_scheme(dev_data)
    dev_data = convert_dataset(dev_data, word_vocab, tag_vocab, segment_vocab)
    dev_data_batch = Batch(dev_data,500)

    train_data_batch = Batch(train_data, 500)
    # create model
    config = Config()

    # update the config
    if os.path.exists(os.path.join(checkpoint_dir + "config.pkl")):
        config = pickle.load(open(os.path.join(checkpoint_dir + "config.pkl"),'rb'))
    else:
        config.checkpoint_dir = checkpoint_dir
        config.vocab_file = os.path.join(checkpoint_dir,"word.vocab")
        config.num_tags = len(tag_vocab)
        config.segment_vocab_file = os.path.join(checkpoint_dir,"seg.vocab")
        config.tag_vocab_file = os.path.join(checkpoint_dir,"tag.vocab")
        pickle.dump(config, open(os.path.join(config.checkpoint_dir,"config.pkl"),'wb'))

    with tf.Session(config=get_config_proto(log_device_placement=False)) as sess:
        model = NERModel(sess, config)
        model.build()
        model.init()

        print("Config:")
        for k,v in config.__dict__.items():
            print(k,'-',v,sep='\t')

        try:
            model.restore_model()
            print("restored model from checkpoint dir")
        except:
            print("create with fresh parameters")
            pass

        dev_saver = tf.train.Saver(tf.global_variables())

        batch_manager = Batch(train_data,config.batch_size)
        checkpoint_loss = 0.0
        step_time = 0.0
        stats_per_step = 50
        stats_per_eval = 1000
        
        best_dev_f1 = config.best_dev_f1
        best_test_f1 = config.best_test_f1

        should_stop = False
        epoch = 0
        while True:
            epoch += 1
            for batch in batch_manager.next_batch():
                start_time = time.time()
                loss, global_step, lr = model.train_one_batch(*zip(*batch))
                step_time += (time.time() - start_time)
                checkpoint_loss += loss

                if global_step % stats_per_step == 0:
                    print("# epoch - {3}, global step - {0}, step time - {1}, loss - {2}, lr - {4}".format(
                        global_step,step_time/stats_per_step, checkpoint_loss/stats_per_step, epoch, lr))
                    checkpoint_loss = 0.0
                    step_time = 0.0

                if global_step % (5*stats_per_eval) == 0:
                    print("Eval train data")
                    train_f1 = evaluate(model, "train", train_data_batch, word_vocab, tag_vocab)

                if global_step % stats_per_eval == 0 or global_step == config.num_train_steps:
                    words, length, segments, target = batch[0]

                    decode,_ = model.inference([words],[length],[segments])
                    print("Sentence:")
                    print(" ".join(word_vocab[w] for w in words[:length]))
                    print("Gold:")
                    print(" ".join([tag_vocab[t] for t in target[:length]]))
                    print("Predict:")
                    print(" ".join([tag_vocab[p] for p in decode[0][:length]]))

                    print("Eval dev data")
                    dev_f1 = evaluate(model,"dev",dev_data_batch,word_vocab,tag_vocab)
                    print("Eval test data")
                    test_f1 = evaluate(model,'test',test_data_batch,word_vocab,tag_vocab)

                    if dev_f1 > best_dev_f1:
                        best_dev_f1 = dev_f1
                        config.best_dev_f1 = best_dev_f1
                        config.best_test_f1 = test_f1
                        print("New best dev f1 - {0}, test f1 - {1}".format(best_dev_f1, test_f1))
                        dev_saver.save(sess,config.checkpoint_dir + "best_dev/model",global_step)
                        pickle.dump(config, open(os.path.join(config.checkpoint_dir,"config.pkl"),'wb'))

                    if test_f1 > best_test_f1:
                        best_test_f1 = test_f1
                        print("New best test f1 - {0}".format(best_test_f1))

                if global_step >= config.num_train_steps:
                    should_stop = True
                    break

            model.save_model()

            if should_stop:
                print("Best f1 of dev: {0}, test f1: {1}".format(config.best_dev_f1,config.best_test_f1))
                print("Best test f1 is {0}".format(best_test_f1))
                break
