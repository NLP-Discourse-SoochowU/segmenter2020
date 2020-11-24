# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
import torch
import torch.nn as nn
from model.GCN_CRF.crf_layer import CRF_Layer
from config import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model.GCN_CRF.crf_config import *
from model.GCN import GCN


class Segment_Model(nn.Module):
    def __init__(self, word_emb):
        super(Segment_Model, self).__init__()
        # random
        self.word_emb = nn.Embedding(word_emb.shape[0], WORDEMB_SIZE)
        # nre_pre = np.array([arr[0:3] for arr in pretrained])
        self.word_emb.weight.data.copy_(torch.from_numpy(word_emb))
        self.word_emb.weight.requires_grad = True if EMBED_LEARN else False
        self.pos_emb = nn.Embedding(POS_TAG_NUM, POS_TAG_SIZE)
        self.pos_emb.weight.requires_grad = True
        self.sent_encode = nn.LSTM(WORDEMB_SIZE + POS_TAG_SIZE, HIDDEN_SIZE // 2, num_layers=RNN_LAYER,
                                   dropout=DROPOUT, bidirectional=True, batch_first=True)
        self.gcn_s = nn.ModuleList([GCN(HIDDEN_SIZE, HIDDEN_SIZE) for _ in range(GCN_LAYER)])
        self.tagger = nn.Linear(HIDDEN_SIZE, len(TAG_LABELS_))
        self.nnDropout = nn.Dropout(ENC_DEC_DROPOUT)
        self.crf_layer = CRF_Layer()

    def gen_loss(self, inputs, target):
        """ 进行特征抽取和子句分割，直接生成带 CRF 层的损失。
        """
        word_ids, pos_ids, graph, _, _, masks = inputs
        # batch_size, max_seq_len = word_ids.size()
        # (batch_size, seq_len)
        self.sent_encode.flatten_parameters()
        word_emb = self.word_emb(word_ids)
        pos_emb = self.pos_emb(pos_ids)
        rnn_inputs = torch.cat([word_emb, pos_emb], dim=-1)  # (batch_size, padding_length, embedding_size)
        lengths = masks.sum(-1)
        rnn_inputs = rnn_inputs * masks.unsqueeze(-1).float()
        rnn_inputs_packed = pack_padded_sequence(rnn_inputs, lengths, batch_first=True)
        rnn_outputs_packed, _ = self.sent_encode(rnn_inputs_packed)
        rnn_outputs, _ = pad_packed_sequence(rnn_outputs_packed, batch_first=True)
        gcn_outputs = rnn_outputs
        for gcn in self.gcn_s:
            gcn_outputs = gcn(graph, gcn_outputs)
        gcn_feats = self.tagger(gcn_outputs)
        # crf layer, loss computation and crf sorter, tag_score is also the tag rep with gcn features.
        loss_ = self.crf_layer.crf_loss(gcn_feats, target)
        return loss_

    def forward(self, inputs):
        """ 进行特征抽取和子句分割
        """
        word_ids, pos_ids, graph, _, _, masks = inputs
        # batch_size, max_seq_len = word_ids.size()
        # (batch_size, seq_len)
        self.sent_encode.flatten_parameters()
        word_emb = self.word_emb(word_ids)
        pos_emb = self.pos_emb(pos_ids)
        rnn_inputs = torch.cat([word_emb, pos_emb], dim=-1)  # (batch_size, padding_length, embedding_size)
        lengths = masks.sum(-1)
        rnn_inputs = rnn_inputs * masks.unsqueeze(-1).float()
        rnn_inputs_packed = pack_padded_sequence(rnn_inputs, lengths, batch_first=True)
        rnn_outputs_packed, _ = self.sent_encode(rnn_inputs_packed)
        rnn_outputs, _ = pad_packed_sequence(rnn_outputs_packed, batch_first=True)
        gcn_outputs = rnn_outputs
        for gcn in self.gcn_s:
            gcn_outputs = gcn(graph, gcn_outputs)
        gcn_feats = self.tagger(gcn_outputs)
        # crf layer tagging
        score, tag_seq = self.crf_layer(gcn_feats)
        return tag_seq
