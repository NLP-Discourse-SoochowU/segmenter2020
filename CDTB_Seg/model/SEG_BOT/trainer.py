# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
import random

import numpy as np
import torch
import torch.optim as optim
from model.SEG_BOT.model import Segment_Model
from config import *
from model.share import gen_batch_iter, check_max
from model.SEG_BOT.metric import evaluate_
from path_config import IDS2VEC, DATA_SETS, DATA_SETS_SYN, LOG_ALL
from utils.file_util import *

random.seed(SEED)
torch.random.manual_seed(SEED)
np.random.seed(SEED)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main():
    """ Train and evaluate
    """
    log_file = os.path.join(LOG_ALL, "set_" + str(SET) + ".log")
    log_ite = 1
    word_emb = load_data(IDS2VEC)
    model = Segment_Model(word_emb=word_emb)
    if USE_GPU:
        model.cuda(CUDA_ID)
    step, best_f1, max_b, micro_max_, loss_ = 0, 0, 0., 0., 0.
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2)
    train_set, dev_set, test_set = load_data(DATA_SETS_SYN) if USE_ALL_SYN_INFO else load_data(DATA_SETS)
    for epoch in range(1, N_EPOCH):
        batch_iter = gen_batch_iter(train_set)
        for n_batch, (inputs, target) in enumerate(batch_iter, start=1):
            step += 1
            model.train()
            optimizer.zero_grad()
            tag_outputs, targets = target
            loss_ = loss_ + model.gen_loss(inputs, targets)
            if n_batch > 0 and n_batch % LOG_EVE == 0:
                loss_ = loss_ / float(LOG_EVE * BATCH_SIZE)
                loss_.backward()
                optimizer.step()
                print("VERSION: " + str(VERSION) + ", SET: " + str(SET) + " -- lr %f, epoch %d, batch %d, loss %.4f" %
                      (get_lr(optimizer), epoch, n_batch, loss_.item()))
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
    result = evaluate_(batch_iter, model, write_out=write_out, last_max=last_max)
    return result
