# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
import torch
from config import *
import numpy as np
import torch.nn.functional as func


def evaluate_enc_dec(batch_iter, model):
    """ To evaluate the model in EDU segmentation.
        因内存原因，我们依旧选择批量运算的方式
        不同的解码评测，不同的是，这个模式下只要一句一句进行，因为边界是未知
        的，所以模型要做完边界预测才知道下一步做什么。
    """
    c_b, g_b, h_b = 0., 0., 0.
    for n_batch, (inputs, target) in enumerate(batch_iter, start=1):
        word_ids, pos_ids, graph, masks = inputs
        # (batch_size, boundary_num + 1),
        pred = model.decode_all(word_ids, pos_ids, graph, masks)
        # (batch_size, boundary_num + 1, seq_len) --> (batch, boundary_num + 1)
        predict = pred.data.cpu().numpy()
        target = target.data.cpu().numpy()
        # 标准标签中标签值为1的下标对应的预测标签值数组中预测正确为1的个数。
        c_b += len(np.intersect1d(predict, target))
        # 人工标注的标签中值为1的标签整体
        g_b += len(target)
        # 预测结果中标签值为1的标签个数
        h_b += len(predict)
        if LearnFromEnd:
            # 因为均多包含了句子结束符，需要减去
            c_b -= 1
            g_b -= 1
            h_b -= 1
    # p r f
    p_b = 0. if h_b == 0 else c_b / h_b
    r_b = 0. if g_b == 0 else c_b / g_b
    f_b = 0. if (g_b + h_b) == 0 else (2 * c_b) / (g_b + h_b)
    return p_b, r_b, f_b


def evaluate_ori(batch_iter, model):
    """ To evaluate the model in EDU segmentation.
        因内存原因，我们依旧选择批量运算的方式
    """
    c_b, g_b, h_b = 0., 0., 0.
    # c_c, g_c, h_c = 0., 0., 0.
    for n_batch, (inputs, target_) in enumerate(batch_iter, start=1):
        word_ids, pos_ids, graph, decode_indices, decode_mask, masks = inputs
        target, _ = target_
        # (batch_size, seq_len, label_size)
        score_ = model(word_ids, pos_ids, graph, masks, decode_mask, decode_indices)
        # (batch, seq_len, label_size) --> (batch, seq_len)
        pred = torch.argmax(func.log_softmax(score_, dim=-1), dim=2)
        pred = pred[:, 1:]
        target = target[:, 1:]
        masks = masks[:, 1:]
        # scoring
        predict = pred.mul(masks.long()).view(-1).data.cpu().numpy()
        target = target.contiguous().view(-1).data.cpu().numpy()
        trg_idx = np.argwhere(target == 1).reshape(-1)
        b_pred = predict[trg_idx]
        # 标准标签中标签值为1的下标对应的预测标签值数组中预测正确为1的个数。
        c_b += sum(b_pred)
        # 人工标注的标签中值为1的标签整体
        g_b += trg_idx.shape[0]
        # 预测结果中标签值为1的标签个数
        h_b += sum(predict)
    # p r f
    p_b = 0. if h_b == 0 else c_b / h_b
    r_b = 0. if g_b == 0 else c_b / g_b
    f_b = 0. if (g_b + h_b) == 0 else (2 * c_b) / (g_b + h_b)
    return p_b, r_b, f_b

