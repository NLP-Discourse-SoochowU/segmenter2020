# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
import torch
import torch.nn as nn
from model.GCN import GCN
import torch.nn.functional as func
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import *


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
        if RNN_TYPE == "LSTM":
            self.sent_encode = nn.LSTM(WORDEMB_SIZE + POS_TAG_SIZE, HIDDEN_SIZE // 2, num_layers=RNN_LAYER,
                                       dropout=DROPOUT, bidirectional=True, batch_first=True)
        else:
            self.sent_encode = nn.GRU(WORDEMB_SIZE + POS_TAG_SIZE, HIDDEN_SIZE // 2, num_layers=RNN_LAYER,
                                      dropout=DROPOUT, bidirectional=True, batch_first=True)
        self.gcn_s = nn.ModuleList([GCN(HIDDEN_SIZE, HIDDEN_SIZE) for _ in range(GCN_LAYER)])
        self.tagger = nn.Linear(HIDDEN_SIZE, len(TAG_LABELS))

    def gen_loss(self, inputs, tag_outputs):
        word_ids, pos_ids, graph, decode_indices, decode_mask, masks = inputs
        batch_size, max_seq_len = word_ids.size()
        self.sent_encode.flatten_parameters()
        word_emb = self.word_emb(word_ids)
        pos_emb = self.pos_emb(pos_ids)
        rnn_inputs = torch.cat([word_emb, pos_emb], dim=-1)  # (batch_size, padding_length, embedding_size)
        if masks is not None:
            lengths = masks.sum(-1)
            rnn_inputs = rnn_inputs * masks.unsqueeze(-1).float()
            rnn_inputs_packed = pack_padded_sequence(rnn_inputs, lengths, batch_first=True)
            rnn_outputs_packed, _ = self.sent_encode(rnn_inputs_packed)
            rnn_outputs, _ = pad_packed_sequence(rnn_outputs_packed, batch_first=True)
        else:
            rnn_outputs, _ = self.sent_encode(rnn_inputs)
        # print("(batch_size, seq_len, hidden): ", rnn_outputs.size())
        gcn_outputs = rnn_outputs
        for gcn in self.gcn_s:
            gcn_outputs = gcn(graph, gcn_outputs)
        tag_score = self.tagger(gcn_outputs)
        pred = func.log_softmax(tag_score, dim=-1)
        pred = pred.view(batch_size * max_seq_len, -1)
        target = tag_outputs.view(-1)
        masks = masks.view(-1)
        losses = func.nll_loss(pred, target, reduction='none')
        loss_ = (losses * masks.float()).sum() / masks.sum().float()
        return loss_

    def forward(self, inputs=None):
        word_ids, pos_ids, graph, decode_indices, decode_mask, masks = inputs
        self.sent_encode.flatten_parameters()
        word_emb = self.word_emb(word_ids)
        pos_emb = self.pos_emb(pos_ids)
        rnn_inputs = torch.cat([word_emb, pos_emb], dim=-1)  # (batch_size, padding_length, embedding_size)
        lengths = masks.sum(-1)
        rnn_inputs = rnn_inputs * masks.unsqueeze(-1).float()
        rnn_inputs_packed = pack_padded_sequence(rnn_inputs, lengths, batch_first=True)
        rnn_outputs_packed, _ = self.sent_encode(rnn_inputs_packed)
        rnn_outputs, _ = pad_packed_sequence(rnn_outputs_packed, batch_first=True)
        # print("(batch_size, seq_len, hidden): ", rnn_outputs.size())
        gcn_outputs = rnn_outputs
        for gcn in self.gcn_s:
            gcn_outputs = gcn(graph, gcn_outputs)
        tag_score = self.tagger(gcn_outputs)
        return tag_score
