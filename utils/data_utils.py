import re
import codecs
from utils.model_utils import PAD, PAD_ID, UNK, UNK_ID
from collections import Counter
import jieba
import random
import math

def add_external_words(fname):
    '''
    Add external words to jieba dict
    each line contains one word
    '''
    cnt = 0
    with open(fname,encoding='utf8') as f:
        for line in f:
            word = line.strip()
            cnt += 1
            jieba.add_word(word)
    print("add {0} words from {1}".format(cnt,fname))

def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags

def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags

def read_data(fnames, zeros=False, lower=False):
    '''
    Read all data into memory and convert to iobes tags.
    A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    Args:
        - fnames: a list of filenames contain the data
        - zeros: if we need to replace digits to 0s
    Return:
        - sentences: a list of sentnences, each sentence contains a list of (word,tag) pairs
    '''
    sentences = []
    sentence = []
    if not isinstance(fnames, list):
        fnames = [fnames]
    for fname in fnames:
        sentence_num = 0
        num = 0
        print("read data from file {0}".format(fname))
        for line in codecs.open(fname, 'r', 'utf8'):
            num+=1
            line = line.rstrip()
            line = re.sub("\d+",'0',line) if zeros else line
            if not line:
                if len(sentence) > 0:
                    sentences.append(sentence)
                    sentence_num += 1
                    sentence = []
            else:
                # in case space is a word
                if line[0] == " ":
                    line = "$" + line[1:]
                    word = line.split()
                else:
                    word= line.split(' ')
                assert len(word) >= 2, print(fname,num,[word[0]],line)
                word[0] = word[0].lower() if lower else word[0]
                sentence.append(word)
        if len(sentence) > 0:
            sentence_num += 1
            sentences.append(sentence)
        print("Got {0} sentences from file {1}".format(sentence_num,fname))
    print("Read all the sentences from training files: {0} sentences".format(len(sentences)))
    return sentences

def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True

def update_tag_scheme(sentences, tag_scheme='iobes'):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            # we already did that in iob2 method
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')

def create_vocab(sentences, lower_case=False, min_cnt = 0):
    print("Create vocab with lower case: {0}, min count: {1}".format(lower_case,min_cnt))
    word_count = Counter()
    tag_count = Counter()
    for sentence in sentences:
        ws,t = zip(*sentence)
        word_count.update([w.lower() if lower_case else w for w in ws])
        tag_count.update(t)
    word_vocab = [PAD,UNK]
    tag_vocab = []
    for w,c in word_count.most_common():
        if c < min_cnt:
            break
        word_vocab.append(w)
    for t,c in tag_count.most_common():
        tag_vocab.append(t)
    print("word vocab size: {0}, tag vocab size: {1}".format(len(word_vocab),len(tag_vocab)))
    return word_vocab,tag_vocab

def save_vocab(vocab, filename):
    with codecs.open(filename,'w','utf8') as f:
        for w in vocab:
            f.write(w + "\n")

def segment_vocab():
    '''
    BIES for segement
    '''
    return ['S','B','I','E']

def get_segment_tags(sentence):
    tags = []
    for word in jieba.cut(sentence):
        if len(word) == 1:
            tags.append('S')
        else:
            t = ['I'] * len(word)
            t[0] = 'B'
            t[-1] = 'E'
            tags.extend(t)
    return tags

def result_to_json(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
        elif tag[0] == "B":
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item

def convert_sentence(sentence, word2id, segment2id):
    sentence = sentence.replace(" ",'')
    segment_tags = get_segment_tags(sentence)
    words = [word2id[w] if w in word2id else word2id[UNK] for w in sentence]
    segment_tags = [segment2id[s] for s in segment_tags]
    return words, len(words), segment_tags

def convert_dataset(sentences, word_vocab, tag_vocab, segment_vocab):
    word2id = {w:i for i,w in enumerate(word_vocab)}
    tag2id = {t:i for i,t in enumerate(tag_vocab)}
    segment2id = {s:i for i,s in enumerate(segment_vocab)}
    
    finals = []
    for sentence in sentences:
        words,tags = zip(*sentence)
        segment_tags = get_segment_tags("".join(words))
        words = [word2id[w] if w in word2id else word2id[UNK] for w in words]
        tags = [tag2id[t] for t in tags]
        segment_tags = [segment2id[s] for s in segment_tags]
        finals.append((words,len(words),tags,segment_tags))
    return finals

class Batch(object):
    def __init__(self, sentences, batch_size = 20):
        self.data_size = len(sentences)
        self.batch_size = batch_size
        self.num_batch = int(math.ceil(self.data_size / self.batch_size))

        # sort sentences by length
        self.sentences = sorted(sentences,key=lambda item: item[1])
        self.batch_data = self.patch_to_batches()

    def patch_to_batches(self):
        batch_data = list()
        for i in range(self.num_batch):
            batch_data.append(self.pad_data(self.sentences[i*self.batch_size : (i+1)*self.batch_size]))
        return batch_data
    
    def pad_data(self, sentences):
        max_length = max([sentence[1] for sentence in sentences])
        padded_sentences = []
        for sentence in sentences:
            words, length, tags, segs = sentence
            padding = [PAD_ID] * (max_length - length)
            padded_sentences.append((words + padding,length,segs+padding,tags+padding))
        return padded_sentences
        
    def next_batch(self, shuffle = True):
        if shuffle:
            random.shuffle(self.batch_data)
        for batch in self.batch_data:
            yield batch