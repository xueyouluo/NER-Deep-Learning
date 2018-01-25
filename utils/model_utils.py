import os
import codecs
import csv 
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np

UNK = '<unk>'
UNK_ID = 0
PAD = '<pad>'
PAD_ID = 1

def get_optimizer(opt):
    """
    A function to get optimizer.

    :param opt: optimizer function name
    :returns: the optimizer function
    :raises assert error: raises an assert error
    """
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert False
    return optfn

def load_embed_txt(embed_file):
    """Load embed_file into a python dictionary.
    Note: the embed_file should be a Glove formated txt file. Assuming
    embed_size=5, for example:
    the -0.071549 0.093459 0.023738 -0.090339 0.056123
    to 0.57346 0.5417 -0.23477 -0.3624 0.4037
    and 0.20327 0.47348 0.050877 0.002103 0.060547
    Args:
        embed_file: file path to the embedding file.
    Returns:
        a dictionary that maps word to vector, and the size of embedding dimensions.
    """
    emb_dict = dict()
    emb_size = None
    with codecs.getreader("utf-8")(tf.gfile.GFile(embed_file, 'rb')) as f:
        for line in f:
            tokens = line.strip().split(" ")
            word = tokens[0]
            vec = list(map(float, tokens[1:]))
            emb_dict[word] = vec
            if emb_size:
                assert emb_size == len(vec), "All embedding size should be same."
            else:
                emb_size = len(vec)
    return emb_dict, emb_size

def load_pretrained_emb_from_txt(id2word, embed_file):
    '''
    load pretrained embedding from txt
    '''
    print("load pretrained embedding from {0}...".format(embed_file))

    emb_dict, emb_size = load_embed_txt(embed_file)
    embedding = np.zeros([len(id2word),emb_size],dtype=np.float32)
    in_cnt = 0
    lower_cnt = 0
    not_in_cnt = 0
    for idx,token in id2word.items():
        if token in emb_dict:
            embedding[idx] = emb_dict[token]
            in_cnt += 1
        elif token.lower() in emb_dict:
            embedding[idx] = emb_dict[token.lower()]
            lower_cnt += 1
        else:
            not_in_cnt += 1
    print("vocab size: {0}, embbeding vocab size:{1}, embbedding dim: {2} \nfound {3} words, {4} words lower case, {5} not in embbeding vocab ".format(
        len(id2word),len(emb_dict),emb_size,in_cnt,lower_cnt,not_in_cnt
    ))
    return embedding    

def read_tag_vocab(tag_file):
    vocab = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(tag_file, "rb")) as f:
        vocab_size = 0
        for word in f:
            vocab_size += 1
            vocab.append(word.strip())
    word2id = {}
    for word in vocab:
        word2id[word] = len(word2id)
    id2word = {i: w for w, i in word2id.items()}
    return word2id,id2word  

def read_vocab(vocab_file):
    """read vocab from file, one word per line
    """
    vocab = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
      vocab_size = 0
      for word in f:
        vocab_size += 1
        vocab.append(word.strip())
    
    if vocab[1] != UNK or vocab[0] != PAD:
        print("The first vocab word %s %s"
                    " is not %s %s" %
                    (vocab[0],vocab[1], PAD, UNK))
        vocab = [UNK,PAD] + vocab
        vocab_size += 2
    
    word2id = {}
    for word in vocab:
        word2id[word] = len(word2id)
    id2word = {i: w for w, i in word2id.items()}
    return word2id,id2word  

def write_metadata(fpath, id2word):
    """Writes metadata file for Tensorboard word embedding visualizer as described here:
    https://www.tensorflow.org/get_started/embedding_viz
    Args:
    fpath: place to write the metadata file
    """
    print("Writing word embedding metadata file to {0}...".format(fpath))
    with open(fpath, "w") as f:
        fieldnames = ['word']
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        for i in range(len(id2word)):
            writer.writerow({"word": id2word[i]})

def add_emb_vis(embedding_var, id2word, checkpoint_dir):
    """Do setup so that we can view word embedding visualization in Tensorboard, as described here:
    https://www.tensorflow.org/get_started/embedding_viz
    Make the vocab metadata file, then make the projector config file pointing to it."""
    print("add embedding to tensorboard")
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    vocab_metadata_path = os.path.join(checkpoint_dir, "vocab_metadata.tsv")
    write_metadata(vocab_metadata_path, id2word) # write metadata file
    summary_writer = tf.summary.FileWriter(checkpoint_dir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = vocab_metadata_path
    projector.visualize_embeddings(summary_writer, config)