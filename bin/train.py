from model.model import NERModel
from model.config import Config
from utils.data_utils import read_data, update_tag_scheme, create_vocab, save_vocab, segment_vocab, convert_dataset, Batch
import os
from utils.train_utils import  get_config_proto
import tensorflow as tf
import time
from utils.evaluate import evaluate

if __name__ == "__main__":
    DATA_DIR = "./data"
    
    # read training data
    train_data = read_data(os.path.join(DATA_DIR,"example.train"))

    # convert tags to iobes
    update_tag_scheme(train_data)

    # create vocab from training data
    word_vocab,tag_vocab = create_vocab(train_data)
    segment_vocab = segment_vocab()

    # save vocab
    save_vocab(word_vocab, os.path.join(DATA_DIR,"word.vocab"))
    save_vocab(tag_vocab,os.path.join(DATA_DIR,"tag.vocab"))
    save_vocab(segment_vocab,os.path.join(DATA_DIR,"seg.vocab"))

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
    config.vocab_file = os.path.join(DATA_DIR,"word.vocab")
    config.pretrained_embedding_file = os.path.join(DATA_DIR,"wiki_100.utf8")
    config.num_tags = len(tag_vocab)

    with tf.Session(config=get_config_proto()) as sess:
        model = NERModel(sess, config)
        model.build()
        model.init()

        try:
            model.restore_model()
            print("restored model from checkpoint dir")
        except:
            pass
        batch_manager = Batch(train_data,64)
        checkpoint_loss = 0.0
        step_time = 0.0
        stats_per_step = 100
        best_dev_f1 = -100
        best_test_f1 = -100

        for i in range(20):
            for batch in batch_manager.next_batch():
                start_time = time.time()
                loss, global_step = model.train_one_batch(*zip(*batch))
                step_time += (time.time() - start_time)
                checkpoint_loss += loss

                if global_step % stats_per_step == 0:
                    print("# global step - {0}, step time - {1}, loss - {2}".format(global_step,step_time/stats_per_step, checkpoint_loss/stats_per_step))
                    checkpoint_loss = 0.0
                    step_time = 0.0

            dev_f1 = evaluate(model,"dev",dev_data,word_vocab,tag_vocab)
            test_f1 = evaluate(model,'test',test_data,word_vocab,tag_vocab)

            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                print("New best dev f1 - {0}".format(best_dev_f1))
                model.save_model(config.checkpoint_dir + "/best_dev")

            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                print("New best test f1 - {0}".format(best_test_f1))
                model.save_model(config.checkpoint_dir + "/best_test")

            model.save_model()


