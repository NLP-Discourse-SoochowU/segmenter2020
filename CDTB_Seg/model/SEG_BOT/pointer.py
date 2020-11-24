# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
from config import *
import torch.nn as nn
import torch.nn.functional as func


class Pointer(nn.Module):
    def __init__(self, encoder_size, decoder_size, num_labels=1, hidden_size=HIDDEN_SIZE):
        super(Pointer, self).__init__()
        self.nnSELU = nn.SELU()
        self.nnW1 = nn.Linear(encoder_size, hidden_size, bias=False)
        self.nnW2 = nn.Linear(decoder_size, hidden_size, bias=False)
        self.nnV = nn.Linear(HIDDEN_SIZE, 1, bias=False)

    def forward(self, en, di):
        """ param en:  [batch, seq_len, HIDDEN]
            param di:  [batch, num_boundary, HIDDEN]
        """

        batch, seq_len, hidden = en.size()
        batch, num_boundary, hidden = di.size()
        we = self.nnW1(en)  # (batch, seq_len, hidden)
        # di  [batch, num_boundary, HIDDEN] --> [batch, num_boundary, seq_len, HIDDEN]
        di = di.unsqueeze(2)
        di = di.permute(2, 0, 1, 3)
        # (batch, num_boundary, seq_len, hidden)
        exdi = di.expand(seq_len, batch, num_boundary, hidden).permute(1, 2, 0, 3)
        wd = self.nnW2(exdi)  # (batch, num_boundary, seq_len, hidden)
        att_weights = self.nnV(self.nnSELU(we + wd))
        # nn_v = nn_v.permute(3, 2)  # (batch, num_boundary, 1, seq_len)
        # nn_v = self.nnSELU(nn_v)
        # att_weights = func.softmax(nn_v, dim=-1)
        return att_weights
