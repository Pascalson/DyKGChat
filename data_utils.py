# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import random
import importlib

from tensorflow.python.platform import gfile
import tensorflow as tf

from knowledge_graph import *
import args
FLAGS = args.FLAGS

# Special vocabulary symbols
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_KB = b"_KB"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK, _KB]

read_kb = importlib.import_module('.'.join(FLAGS.data_dir.split('/'))+'.read_kb')
KG = KGraph(FLAGS.data_dir, lambda x: read_kb.read_in_graph(x))
nodes = KG.get_vocab_nodes()
str_nodes = KG.get_nodes()
edge_types = KG.get_edge_types()
print(edge_types)
print(len(nodes))

kbstart = 0
kbend = len(nodes) - 1
kdim = len(nodes)
edim = len(edge_types)
print(edim)

################################
import numpy as np

def load_embed(f):
    s_arr = [line.strip().split('\t') for line in f.readlines()]
    vec_arr = [[float(s) for s in line[1][1:-1].split(', ')] for line in s_arr]
    name_arr = [line[0] for line in s_arr]
    embed_dict = {}
    for name, vec in zip(name_arr, vec_arr):
        embed_dict[name] = vec
    return embed_dict

with open(FLAGS.data_dir+'/transE/entityVector.txt', 'r') as fn, \
    open(FLAGS.data_dir+'/transE/relationVector.txt', 'r') as fe:
    node_dict = load_embed(fn)
    edge_type_dict = load_embed(fe)

kbembed_size = len(node_dict[str_nodes[0]])*3
triple_num = FLAGS.fact_size

################################

PAD_ID = 0 + len(nodes)
GO_ID = 1 + len(nodes)
EOS_ID = 2 + len(nodes)
UNK_ID = 3 + len(nodes)
KB_ID = 4 + len(nodes)

# Regular expressions used to tokenize.
#_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        """Plz costomize the tokenizer for different dataset"""
        #TODO words.extend(_WORD_SPLIT.split(space_separated_fragment))
        words.append(space_separated_fragment)
    return [w for w in words if w]

def get_split_path(data_path):
    data_dir, file_name = data_path.rsplit('/',1)
    train_path = os.path.join(data_dir, 'train_' + file_name)
    dev_path = os.path.join(data_dir, 'dev_' + file_name)
    if not (gfile.Exists(train_path)) and ( not gfile.Exists(dev_path) ):
        if not (gfile.Exists(data_path)):
            raise ValueError("Source file %s not found.", data_path)
        raise ValueError("Train file or development file not found.")
    return (train_path, dev_path)

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)
                line = tf.compat.as_bytes(line)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                    if word in vocab and word not in nodes:
                        vocab[word] += 1
                    elif word not in nodes:
                        vocab[word] = 1
        vocab_list = nodes + _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")

def initialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip().encode('utf-8') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        print(len(vocab))
        print(len(rev_vocab))
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]

def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab,
                                                tokenizer, normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_data(data_dir, data_path, vocabulary_size, tokenizer=None):
    train_path, dev_path = get_split_path(data_path)
    vocab_path = os.path.join(data_dir, "vocab%d" % vocabulary_size)
    create_vocabulary(vocab_path, train_path, vocabulary_size, tokenizer, normalize_digits=False)
    train_ids_path = train_path + (".ids%d" % vocabulary_size)
    data_to_token_ids(train_path, train_ids_path, vocab_path, tokenizer, normalize_digits=False)
    dev_ids_path = dev_path + (".ids%d" % vocabulary_size)
    data_to_token_ids(dev_path, dev_ids_path, vocab_path, tokenizer, normalize_digits=False)
    return (train_ids_path, dev_ids_path, vocab_path)

def prepare_info(data_dir, data_path):
    train_path, dev_path = get_split_path(data_path)
    train_info_path = train_path.rsplit('.',1)[0] + (".info")
    dev_info_path = dev_path.rsplit('.',1)[0] + (".info")
    return (train_info_path, dev_info_path)

def prepare_kb(data_dir, data_path):
    train_path, dev_path = get_split_path(data_path)
    train_kb_path = train_path.rsplit('.',1)[0] + (".sp5")
    dev_kb_path = dev_path.rsplit('.',1)[0] + (".sp5")
    return (train_kb_path, dev_kb_path)

def test_info_data(data_dir, data_path, vocabulary_size, tokenizer=None):
    data_dir, file_name = data_path.rsplit('/',1)
    test_path = os.path.join(data_dir, 'test_' + file_name)
    test_ids_path = test_path + (".ids%d" % vocabulary_size)
    vocab_path = os.path.join(data_dir, "vocab%d" % vocabulary_size)
    data_to_token_ids(test_path, test_ids_path, vocab_path, tokenizer, normalize_digits=False)
    test_info_path = test_path.rsplit('.',1)[0] + (".info")
    test_kb_path = test_path.rsplit('.',1)[0] + (".sp5")
    return (test_ids_path, test_info_path, test_kb_path)
