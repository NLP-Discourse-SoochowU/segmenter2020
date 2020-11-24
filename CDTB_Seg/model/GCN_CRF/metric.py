# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
from model.GCN_CRF.crf_config import *
from path_config import TEST_OUT
import numpy as np
from utils.file_util import write_iterate


def evaluate_(batch_iter, model, write_out=False, last_max=0):
    """ To evaluate the model in EDU segmentation.
    """
    model.eval()
    write_all = []
    c_b, g_b, h_b = 0., 0., 0.
    for n_batch, (inputs, target_) in enumerate(batch_iter, start=1):
        _, _, _, _, _, masks = inputs
        target, _ = target_
        tag_seq = model(inputs)
        pred = tag_seq.unsqueeze(0)
        # scoring
        predict = pred.mul(masks.long()).view(-1).data.cpu().numpy()
        target = target.contiguous().view(-1).data.cpu().numpy()
        # boundary
        trg_idx = np.argwhere(target == 1).reshape(-1)
        b_pred = predict[trg_idx]
        # 标准标签中标签值为1的下标对应的预测标签值数组中预测正确为1的个数。
        c_b += sum(b_pred)
        # 人工标注的标签中值为1的标签整体
        g_b += trg_idx.shape[0]
        # 预测结果中标签值为1的标签个数
        h_b += sum(predict)
        if write_out:
            predict_ = [ids2tag_[idx] for idx in predict.tolist()]
            write_all.append(" ".join(predict_))
    # p r f
    p_b = 0. if h_b == 0 else c_b / h_b
    r_b = 0. if g_b == 0 else c_b / g_b
    f_b = 0. if (g_b + h_b) == 0 else (2 * c_b) / (g_b + h_b)
    if write_out and f_b > last_max:
        write_iterate(write_all, TEST_OUT)
    return p_b, r_b, f_b
