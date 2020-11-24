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
from path_config import IDS2VEC, DATA_SETS, DATA_SETS_SYN, LOG_ALL
import torch
from model.share import gen_batch_iter, check_max
from utils.file_util import *
from model.GCN_basic.model import Segment_Model
import torch.optim as optim
from model.GCN_basic.metric import evaluate_

random.seed(SEED)
torch.random.manual_seed(SEED)
np.random.seed(SEED)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def gen_loss(inputs, target, model):
    """ 进行特征抽取和子句分割
    """
    word_ids, pos_ids, graph, decode_indices, decode_mask, masks = inputs
    batch_size, max_seq_len = word_ids.size()
    # (batch_size, seq_len)
    score_ = model(inputs)
    pred = func.log_softmax(score_, dim=-1)
    pred = pred.view(batch_size * max_seq_len, -1)
    target = target.view(-1)
    masks = masks.view(-1)
    losses = func.nll_loss(pred, target, reduction='none')
    loss_ = (losses * masks.float()).sum() / masks.sum().float()
    return loss_


def main():
    """ Train and evaluate
    """
    log_file = os.path.join(LOG_ALL, "set_" + str(SET) + ".log")
    log_ite = 1
    word_emb = load_data(IDS2VEC)
    model = Segment_Model(word_emb=word_emb)
    if USE_GPU:
        model.cuda(CUDA_ID)
    # train
    step, best_f1, max_b, micro_max_, loss_ = 0, 0, 0., 0., 0.
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2)
    train_set, dev_set, test_set = load_data(DATA_SETS_SYN) if USE_ALL_SYN_INFO else load_data(DATA_SETS)
    for epoch in range(1, N_EPOCH):
        batch_iter = gen_batch_iter(train_set)
        for n_batch, (inputs, target) in enumerate(batch_iter, start=1):
            step += 1
            model.train()
            optimizer.zero_grad()
            tag_outputs, _ = target
            loss_ = loss_ + model.gen_loss(inputs, tag_outputs)
            if n_batch > 0 and n_batch % LOG_EVE == 0:
                loss_ = loss_ / float(LOG_EVE * BATCH_SIZE)
                loss_.backward()
                optimizer.step()
                # print("VERSION: " + str(VERSION) + ", SET: " + str(SET) + " -- lr %f, epoch %d, batch %d, loss %.4f" %
                #       (get_lr(optimizer), epoch, n_batch, loss_.item()))
                loss_ = 0.
                if n_batch % EVA_EVE == 0:
                    p_b, r_b, f_b = evaluate(dev_set, model)
                    if f_b > best_f1:
                        best_f1 = f_b
                        p_b_, r_b_, f_b_ = evaluate(test_set, model, write_out=True, permute=False, last_max=micro_max_)
                        if f_b_ > micro_max_:
                            micro_max_ = f_b_
                            max_b = (p_b_, r_b_, f_b_)
                            check_str = check_max(max_b, n_epoch=epoch)
                            print_eve(ite=log_ite, str_=check_str, log_file=log_file)
                            log_ite += 1


def evaluate(dataset, model, write_out=False, permute=True, last_max=0):
    model.eval()
    batch_iter = gen_batch_iter(dataset, permute=permute)
    # baseline 和 非编码解码结构 采用相同的评测脚本
    result = evaluate_(batch_iter, model, write_out=write_out, last_max=last_max)
    return result
