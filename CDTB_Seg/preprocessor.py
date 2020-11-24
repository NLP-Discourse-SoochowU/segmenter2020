# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
import torch
import random
import numpy as np
from path_config import *
from config import *
import progressbar
from utils.file_util import *
from model.sentence import Sentence
p = progressbar.ProgressBar()
p_1 = progressbar.ProgressBar()


def build_voc(datasets_raw_):
    """ word2ids, pos2ids save all these dictionaries.
    """
    # 人民日报 words
    words_set = set()
    with open(RM300, "r") as f:
        for line in f:
            tokens = line.split()
            words_set.add(tokens[0])
    # build word2ids
    word2ids, pos2ids, word2freq = dict(), dict(), dict()
    word2ids[PAD], word2ids[UNK] = 0, 1
    pos2ids[PAD], pos2ids[UNK] = 0, 1
    idx_1, idx_2 = 2, 2
    train_set, dev_set, test_set = datasets_raw_
    """ 
        instances.append((sent_words, sent_poses, graph))
        tags.append(sent_tags)
    """
    total_instances = train_set[0] + dev_set[0] + test_set[0]
    for instance in total_instances:
        sent_words, sent_poses, graph = instance
        for word, pos_tag in zip(sent_words, sent_poses):
            if word not in word2freq.keys():
                word2freq[word] = 1
            elif word not in word2ids.keys() and word in words_set:
                word2freq[word] += 1
                word2ids[word] = idx_1
                idx_1 += 1
            else:
                word2freq[word] += 1
            if pos_tag not in pos2ids.keys():
                pos2ids[pos_tag] = idx_2
                idx_2 += 1
    # 低频词过滤
    for word in word2freq.keys():
        if word not in word2ids.keys():
            word2ids[word] = word2ids[UNK]
    save_data(word2ids, WORD2IDS)
    save_data(pos2ids, POS2IDS)
    build_ids2_vec()


def build_ids2_vec():
    """ 构建 ids2vec
    """
    word2ids = load_data(WORD2IDS)
    ids2vec = dict()
    with open(RM300, "r") as f:
        for line in f:
            tokens = line.split()
            word = tokens[0]
            vec = np.array([[float(token) for token in tokens[1:]]])
            if tokens[0] in word2ids.keys() and word2ids[tokens[0]] != UNK_ID:
                ids2vec[word2ids[word]] = vec
    # transform into numpy array
    # PAD and UNK
    embed = [np.zeros(shape=(WORDEMB_SIZE,), dtype=np.float32)]
    embed = np.append(embed, [np.random.uniform(-0.25, 0.25, WORDEMB_SIZE)], axis=0)
    # others
    idx_valid = list(ids2vec.keys())
    idx_valid.sort()
    for idx in idx_valid:
        # print(embed.size())
        # print(len(ids2vec[idx]))
        embed = np.append(embed, ids2vec[idx], axis=0)
    save_data(embed, IDS2VEC)


def build_ids2word():
    word2ids = load_data(WORD2IDS)
    ids2word = dict()
    for key_, val_ in zip(word2ids.keys(), word2ids.values()):
        if val_ == UNK_ID:
            ids2word[val_] = UNK
        else:
            ids2word[val_] = key_
    save_data(ids2word, IDS2WORD)


def build_data_ids(datasets_raw_):
    """ Load sentences and EDUs from source files and generate the data list with ids.
        (word_ids, pos_ids, syn_ids, tag_ids)
    """
    train_set_, dev_set_, test_set_ = datasets_raw_
    word2ids, pos2ids = load_data(WORD2IDS), load_data(POS2IDS)
    # train dev and test sets
    train_list = gen_specific_instances(train_set_, word2ids, pos2ids)
    dev_list = gen_specific_instances(dev_set_, word2ids, pos2ids)
    test_list = gen_specific_instances(test_set_, word2ids, pos2ids)
    data_set = (train_list, dev_list, test_list)
    # save data (ids of all)
    save_data(data_set, DATA_SETS)


def gen_specific_instances(data_set, word2ids, pos2ids):
    """ Transform all data into ids.
        We take root node into consideration. data_set = (sent, pos, tag)
    """
    data_set_ = []
    instances, tags = data_set
    # sents, pos_tags, tags = data_set
    p_2 = progressbar.ProgressBar()
    p_2.start(len(instances))
    p2_idx = 1
    for instance, tag in zip(instances, tags):
        p_2.update(p2_idx)
        p2_idx += 1
        sent_words, sent_poses, graph = instance
        tag_seq = tag
        sent_words_ids = [word2ids[word] for word in sent_words]
        sent_poses_ids = [pos2ids[pos] for pos in sent_poses]
        sent_tags_ids = [tag2ids[tag_] for tag_ in tag_seq]
        # root PAD
        # sent_words_ids.insert(0, PAD_ID)
        # sent_poses_ids.insert(0, PAD_ID)
        # sent_tags_ids.insert(0, 0)  # 放置 0 将 root作为内部节点看待
        # 句法信息
        graph_ids = []
        for graph_edge in graph:
            """ 目前只考虑方向 """
            from_, to_, syn_ = graph_edge
            graph_ids.append((from_, to_, sync2ids[syn_]))
        data_set_.append((sent_words_ids, sent_poses_ids, graph_ids, sent_tags_ids))
    p_2.finish()
    return data_set_


if __name__ == "__main__":
    # build voc (word2ids, pos2ids, tag2ids)
    # print("build voc, step one...")
    datasets_raw = load_data(RAW_DATA_SETS)
    # build_voc(datasets_raw)
    # input("build ids, step two...")
    build_data_ids(datasets_raw)
    # build ids2word
    # build_ids2word()
