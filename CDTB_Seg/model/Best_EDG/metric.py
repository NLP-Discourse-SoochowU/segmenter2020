# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
from config import *
from path_config import TEST_OUT
import numpy as np
from utils.file_util import write_iterate


def evaluate_(batch_iter, model, write_out=False, last_max=0):
    """ 对边界对应下标的评测。
    """
    c_b, g_b, h_b = 0., 0., 0.
    write_all = []
    for n_batch, (inputs, target_) in enumerate(batch_iter, start=1):
        # word_ids, pos_ids, graph, _, _, masks = inputs
        _, target = target_
        pred = model(inputs)
        predict = pred.data.cpu().numpy()
        target = target.data.cpu().numpy()
        # 标准标签中标签值为1的下标对应的预测标签值数组中预测正确为1的个数。
        c_b += len(np.intersect1d(predict, target))
        # 人工标注的标签中值为1的标签整体
        g_b += len(target)
        # 预测结果中标签值为1的标签个数
        h_b += len(predict)
        if LearnFromEnd:  # 因为均多包含了句子结束符，需要减去
            c_b -= 1
            g_b -= 1
            h_b -= 1
        if write_out:
            predict = predict.tolist()
            predict_ = []
            idx_ = 0
            for idx in predict:
                predict_ += ["O" for _ in range(idx_, idx)]
                predict_.append("B")
                idx_ = idx + 1
            predict_.pop()
            predict_.append("O")
            write_all.append(" ".join(predict_))
    # p r f
    p_b = 0. if h_b == 0 else c_b / h_b
    r_b = 0. if g_b == 0 else c_b / g_b
    f_b = 0. if (g_b + h_b) == 0 else (2 * c_b) / (g_b + h_b)
    if write_out and f_b > last_max:
        write_iterate(write_all, TEST_OUT)
    return p_b, r_b, f_b
