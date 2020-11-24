# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
import torch
import numpy as np
from config import *
np.random.seed(SEED)


def data_ids_prep(num_batch, max_seq_len, batch):
    """ Transform all the data into the form of ids.
    """
    word_inputs = np.zeros([num_batch, max_seq_len], dtype=np.long)
    pos_inputs = np.zeros([num_batch, max_seq_len], dtype=np.long)
    graph_inputs = np.zeros([num_batch, max_seq_len, max_seq_len, SYN_SIZE], np.uint8)
    tag_outputs = np.zeros([num_batch, max_seq_len], dtype=np.long)  # 每一句对应的分子句的标签 0 1 1 1 2 1 1 2 ...
    masks = np.zeros([num_batch, max_seq_len], dtype=np.uint8)
    decode_indices, decode_mask = None, None
    targets = None
    for i, (word_ids, pos_ids, graph_ids, tag_ids) in enumerate(batch):
        seq_len = len(word_ids)
        word_inputs[i][:seq_len] = word_ids
        pos_inputs[i][:seq_len] = pos_ids
        tag_outputs[i][:seq_len] = tag_ids
        for x, y, z in graph_ids:
            # Use one-hot vector to represent the connection between nodes, 0 denotes no, 1 refers to yes.
            graph_inputs[i, x, y, z] = 1
        masks[i][:seq_len] = 1
        targets = np.where(tag_outputs[i] == 1)[0]
        if LearnFromEnd:
            targets = np.insert(targets, targets.shape[0], max_seq_len-1)
        decode_indices = np.where(tag_outputs[i] == 1)[0]
        # print(decode_indices)
        decode_indices += 1
        decode_indices = np.insert(decode_indices, 0, 0)  # 将第一个元素插入，作为第一个解码的首位
        decode_mask = np.zeros([decode_indices.shape[0], max_seq_len], dtype=np.uint8)
        for idx in range(decode_indices.shape[0]):
            decode_idx_begin = decode_indices[idx]
            decode_mask[idx][decode_idx_begin:] = 1
        # print(tag_outputs[i])
        # input(decode_indices)
    return word_inputs, pos_inputs, graph_inputs, tag_outputs, decode_indices, decode_mask, targets, masks


def gen_batch_iter(training_set, batch_s=BATCH_SIZE, permute=True):
    """ a batch 2 numpy data.
    """
    random_instances = np.random.permutation(training_set) if permute else np.array(training_set)
    num_instances = len(training_set)
    offset = 0
    while offset < num_instances:
        batch = random_instances[offset: min(num_instances, offset + batch_s)]
        num_batch = batch.shape[0]
        lengths = np.zeros(num_batch, dtype=np.int)
        for i, (word_ids, pos_ids, graph_ids, tag_ids) in enumerate(batch):
            lengths[i] = len(word_ids)
        sort_indices = np.argsort(-lengths)  # 从大到小排列，存放对应下标
        lengths = lengths[sort_indices]
        batch = batch[sort_indices]
        max_seq_len = lengths.max()  # 每个批次里面选择最大值进行 pad
        if max_seq_len >= MAX_SEQ_LEN:
            offset = offset + batch_s
            continue
        word_inputs, pos_inputs, graph_inputs, tag_outputs, decode_indices, decode_mask, targets, masks = \
            data_ids_prep(num_batch, max_seq_len, batch)
        offset = offset + batch_s
        # numpy2torch
        word_inputs = torch.from_numpy(word_inputs).long()
        pos_inputs = torch.from_numpy(pos_inputs).long()
        tag_outputs = torch.from_numpy(tag_outputs).long()
        graph_inputs = torch.from_numpy(graph_inputs).byte()
        decode_indices = torch.from_numpy(decode_indices).long()
        decode_mask = torch.from_numpy(decode_mask).byte()
        targets = torch.from_numpy(targets).long()
        masks = torch.from_numpy(masks).byte()
        if USE_GPU:
            word_inputs = word_inputs.cuda(CUDA_ID)
            pos_inputs = pos_inputs.cuda(CUDA_ID)
            tag_outputs = tag_outputs.cuda(CUDA_ID)
            graph_inputs = graph_inputs.cuda(CUDA_ID)
            decode_indices = decode_indices.cuda(CUDA_ID)
            decode_mask = decode_mask.cuda(CUDA_ID)
            targets = targets.cuda(CUDA_ID)
            masks = masks.cuda(CUDA_ID)
        yield (word_inputs, pos_inputs, graph_inputs, decode_indices, decode_mask, masks), (tag_outputs, targets)


def check_max(max_b, n_epoch=0):
    (p_b, r_b, f_b) = max_b
    check_str = "---" + "VERSION: " + str(VERSION) + ", SET: " + str(SET) + ", EPOCH: " + str(n_epoch) + "---\n" + \
                "TEST (B): " + str(p_b) + "(P), " + str(r_b) + "(R), " + str(f_b) + "(F)\n"
    return check_str
