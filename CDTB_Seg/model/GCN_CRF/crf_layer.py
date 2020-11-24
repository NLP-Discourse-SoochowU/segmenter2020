# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
import torch
from config import HIDDEN_SIZE, CUDA_ID
import torch.nn as nn
from torch.autograd import Variable as Var
from model.GCN_CRF.crf_config import START_TAG, STOP_TAG, tag2ids_


class CRF_Layer(nn.Module):
    """ Resort the feat gen by GCN
    """
    def __init__(self):
        super(CRF_Layer, self).__init__()
        self.tagset_size = len(tag2ids_.keys())
        # Matrix of transition parameters.
        # Entry i,j is the score of transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag2ids_[START_TAG], :] = -10000
        self.transitions.data[:, tag2ids_[STOP_TAG]] = -10000
        self.hidden = (Var(torch.randn(2, 1, HIDDEN_SIZE // 2)),
                       Var(torch.randn(2, 1, HIDDEN_SIZE // 2)))

    def crf_loss(self, feats, tags):
        """ 因为 GCN参数较多，GPU 中计算时 batch_size 只能设置为 1，所以 crf 层进行
            调整时不用考虑 mask 问题。
        """
        feats = feats.squeeze(0)
        tags = tags.squeeze(0)
        forward_score = self.emission_score(feats)
        gold_score = self.seq_tag_score(feats, tags)
        crf_loss_ = forward_score - gold_score
        return crf_loss_

    def emission_score(self, feats):
        """ Scoring Function, feats refers to the sentence representation of a sequence.
            feats: (batch_size, seq_len, label_size)
            Do the forward algorithm to compute the partition function (1, 5)
        """
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.).cuda(CUDA_ID)
        init_alphas[0][tag2ids_[START_TAG]] = 0.  # START_TAG has all of the score.
        forward_var = Var(init_alphas)
        # iterate through the gcn feat of a sequence
        for feat in feats:
            alphas_t = None
            for next_tag in range(self.tagset_size):
                # broadcast the emission score（发散分数）: it is the same regardless of the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # (1, 5), the ith entry of trans_score is the score of transitioning to next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the edge (i -> next_tag)
                # before we do log-sum-exp
                # 第一次迭代时理解：
                #   trans_score：所有其他标签到第一个标签（B）的概率；
                #   emit_score：由 lstm 运行进入隐层再到输出层得到标签 Ｂ 的概率；
                #   forward_var：需要学习的转移概率，随机初始化
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the scores.
                log_sum = self.log_sum_exp(next_tag_var).unsqueeze(0)
                alphas_t = log_sum if alphas_t is None else torch.cat((alphas_t, log_sum))
            forward_var = alphas_t.view(1, -1)  # 到第（t-1）step 时５个标签的各自分数
        terminal_var = forward_var + self.transitions[tag2ids_[STOP_TAG]]
        alpha = self.log_sum_exp(terminal_var)
        return alpha

    @staticmethod
    def log_sum_exp(vec):
        """ Compute log sum exp in a numerically stable way for the forward algorithm
            vec 是 1*5, type 是 Variable
        """
        max_score = vec[0, torch.argmax(vec)]
        # max_score维度是１，　max_score.view(1,-1)维度是１＊１，max_score.view(1, -1).expand(1, vec.size()[1])的维度是１＊５
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])  # vec.size()维度是1*5
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))  # 为什么指数之后再求和，而后才log呢

    def seq_tag_score(self, feats, tags):
        """ 得到 gold_seq tag 的 score
        """
        # Gives the score of a provided tag sequence
        score = Var(torch.Tensor([0]).cuda(CUDA_ID))
        tags = torch.cat([torch.LongTensor([tag2ids_[START_TAG]]).cuda(CUDA_ID), tags])
        for i, feat in enumerate(feats):
            # self.transitions[tags[i + 1], tags[i]] 实际得到的是从标签i到标签i+1的转移概率
            # feat[tags[i+1]]
            # feat 是 step i 的输出结果，有５个值，对应 B, I, E, START_TAG, END_TAG, 取对应标签的值
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[tag2ids_[STOP_TAG], tags[-1]]
        return score

    def forward(self, feats):
        """ Convert feat gen by GCN 2 CRF output
            feats: (batch_size, seq_len, label_size)
            output: (batch_size, seq_len, label_size)
            Find the best path, given the features.
        """
        feats = feats.squeeze(0)
        score, tag_seq = self.vite_decode(feats)
        return score, tag_seq

    def vite_decode(self, feats):
        """ 解码，得到预测的序列，以及预测序列的得分
        """
        back_pointers = []
        # Initialize the viterbi variables in log space
        init_vars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vars[0][tag2ids_[START_TAG]] = 0
        forward_var = Var(init_vars.cuda(CUDA_ID))
        # forward_var at step i holds the viterbi variables for step i-1
        for feat in feats:
            bp_t = []  # holds the back_pointers for this step
            vit_vars_t = None  # holds the viterbi variables for this step
            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the previous step,
                # plus the score of transitioning from tag i to next_tag. We don't include
                # the emission scores here because the max does not depend on them (we add them in below)
                # self.transitions[next_tag] means 其他标签（B, I, E, Start, End）到标签 next_tag 的概率
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = torch.argmax(next_tag_var).cpu().item()
                bp_t.append(best_tag_id)
                next_tv = next_tag_var[0][best_tag_id].unsqueeze(0)
                vit_vars_t = next_tv if vit_vars_t is None else torch.cat((vit_vars_t, next_tv))
            # Now add in the emission scores, and assign forward_var to the set of viterbi
            # variables we just computed
            # 从 step0 到 step(i-1) 时5个序列中每个序列的最大 score
            forward_var = (vit_vars_t + feat).view(1, -1)
            back_pointers.append(bp_t)  # bp_t 有５个元素
        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[tag2ids_[STOP_TAG]]  # 其他标签到 STOP_TAG 的转移概率
        best_tag_id = torch.argmax(terminal_var).cpu().item()
        path_score = terminal_var[0][best_tag_id]
        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bp_t in reversed(back_pointers):  # 从后向前走，找到一个best路径
            best_tag_id = bp_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we do not want to return that to the caller)
        start = best_path.pop()
        assert start == tag2ids_[START_TAG]  # Sanity check
        best_path.reverse()
        best_path = torch.Tensor(best_path).long().cuda(CUDA_ID)  # 把从后向前的路径正过来
        return path_score, best_path
