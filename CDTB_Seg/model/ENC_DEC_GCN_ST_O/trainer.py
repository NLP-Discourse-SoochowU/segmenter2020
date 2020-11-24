# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
import random
import torch.nn.functional as func
import numpy as np
from config import *
from path_config import IDS2VEC, DATA_SETS, DATA_SETS_SYN, LOG_ALL, CH_MODEL_SAVE
import torch
from utils.file_util import *
from model.ENC_DEC_GCN_ST_O.model_stronger import Segment_Model
import torch.optim as optim
from model.ENC_DEC_GCN_ST_O.metric import evaluate_enc_dec

random.seed(SEED)
torch.random.manual_seed(SEED)
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


def gen_batch_iter(training_set, batch_s=BATCH_SIZE):
    """ a batch 2 numpy data.
    """
    random_instances = np.random.permutation(training_set)
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
            # GPU limitation
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


def data_ids_prep_eva(num_batch, max_seq_len, batch):
    """ Transform all the data into the form of ids.
    """
    word_inputs = np.zeros([num_batch, max_seq_len], dtype=np.long)
    pos_inputs = np.zeros([num_batch, max_seq_len], dtype=np.long)
    graph_inputs = np.zeros([num_batch, max_seq_len, max_seq_len, SYN_SIZE], np.uint8)
    tag_outputs = np.zeros([num_batch, max_seq_len], dtype=np.long)  # 每一句对应的分子句的标签 0 1 1 1 2 1 1 2 ...
    masks = np.zeros([num_batch, max_seq_len], dtype=np.uint8)
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
    return word_inputs, pos_inputs, graph_inputs, targets, masks


def gen_eva_batch_iter(training_set, batch_s=BATCH_SIZE):
    """ a batch 2 numpy data.
    """
    random_instances = np.random.permutation(training_set)
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
            # GPU limitation
            offset = offset + batch_s
            continue
        word_inputs, pos_inputs, graph_inputs, targets, masks = data_ids_prep_eva(num_batch, max_seq_len, batch)
        offset = offset + batch_s
        # numpy2torch
        word_inputs = torch.from_numpy(word_inputs).long()
        pos_inputs = torch.from_numpy(pos_inputs).long()
        graph_inputs = torch.from_numpy(graph_inputs).byte()
        targets = torch.from_numpy(targets).long()
        masks = torch.from_numpy(masks).byte()
        if USE_GPU:
            word_inputs = word_inputs.cuda(CUDA_ID)
            pos_inputs = pos_inputs.cuda(CUDA_ID)
            graph_inputs = graph_inputs.cuda(CUDA_ID)
            targets = targets.cuda(CUDA_ID)
            masks = masks.cuda(CUDA_ID)
        yield (word_inputs, pos_inputs, graph_inputs, masks), targets


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def gen_enc_dec_loss(inputs, target, model):
    word_ids, pos_ids, graph, decode_indices, decode_mask, masks = inputs
    # (batch_size, boundary_num + 1, seq_len)
    score_, attn = model(word_ids, pos_ids, graph, masks, decode_mask, decode_indices)
    # score_: (boundary_num + 1, seq_len)  target: (boundary_num + 1, 1)
    score_ = score_.squeeze(0)
    loss_pred = func.nll_loss(score_, target)
    # 两两相似性计算
    loss_difference = 0.
    for idx in range(Head_NUM):
        for c_idx in range(idx + 1, Head_NUM):
            tmp_loss = func.cosine_embedding_loss(attn[idx], attn[c_idx], target=torch.Tensor([-1]).cuda(CUDA_ID))
            loss_difference += tmp_loss
    loss_ = loss_pred + loss_difference
    return loss_


def main():
    """ Train and evaluate
    """
    log_file = os.path.join(LOG_ALL, "set_" + str(SET) + ".log")
    log_ite = 1
    # build model
    word_emb = load_data(IDS2VEC)
    model = Segment_Model(word_emb=word_emb)
    if USE_GPU:
        model.cuda(CUDA_ID)
    # train
    step, best_f1, max_b, micro_max_, loss_ = 0, 0, 0., 0., 0.
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", factor=0.7, patience=17, min_lr=7e-5) if LR_DECAY else None
    train_set, dev_set, test_set = load_data(DATA_SETS_SYN) if USE_ALL_SYN_INFO else load_data(DATA_SETS)
    for epoch in range(1, N_EPOCH):
        batch_iter = gen_batch_iter(train_set)
        for n_batch, (inputs, target) in enumerate(batch_iter, start=1):
            step += 1
            model.train()
            optimizer.zero_grad()
            tag_outputs, targets = target
            loss_ = loss_ + gen_enc_dec_loss(inputs, targets, model)
            if n_batch > 0 and n_batch % LOG_EVE == 0:
                loss_ = loss_ / float(LOG_EVE * BATCH_SIZE)
                loss_.backward()
                optimizer.step()
                print("VERSION: " + str(VERSION) + ", SET: " + str(SET) + " -- lr %f, epoch %d, batch %d, loss %.4f" %
                      (get_lr(optimizer), epoch, n_batch, loss_.item()))
                loss_ = 0.
                if n_batch % EVA_EVE == 0:
                    p_b, r_b, f_b = evaluate(dev_set, model)
                    if LR_DECAY:
                        scheduler.step(f_b)
                    if f_b > best_f1:
                        best_f1 = f_b
                        p_b_, r_b_, f_b_ = evaluate(test_set, model)
                        if f_b_ > micro_max_:
                            micro_max_ = f_b_
                            max_b = (p_b_, r_b_, f_b_)
                            check_str = check_max(max_b, n_epoch=epoch)
                            print_eve(ite=log_ite, str_=check_str, log_file=log_file)
                            log_ite += 1
                            if SAVE_MODEL:
                                torch.save(model, os.path.join(CH_MODEL_SAVE, "CH_" + str(SET) + ".model"))


def check_max(max_b, n_epoch=0):
    (p_b, r_b, f_b) = max_b
    check_str = "---" + "VERSION: " + str(VERSION) + ", SET: " + str(SET) + ", EPOCH: " + str(n_epoch) + "---\n" + \
                "TEST (B): " + str(p_b) + "(P), " + str(r_b) + "(R), " + str(f_b) + "(F)\n"
    return check_str


def evaluate(dataset, model):
    model.eval()
    batch_iter = gen_eva_batch_iter(dataset)
    result = evaluate_enc_dec(batch_iter, model)
    return result
