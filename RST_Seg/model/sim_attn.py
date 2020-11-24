# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
from config import *
import torch
import torch.nn as nn
torch.manual_seed(SEED)


class Sim_Attn(nn.Module):
    def __init__(self, encoder_size, decoder_size, num_labels=1, hidden_size=HIDDEN_SIZE):
        """ num_labels 代表当前节点到各个节点之间做的 bi_affine attention 分配给各个节点的向量维度，
            因为子句分割是分割点选择任务，所以每个分割点的维度设置为1即可，一组标量的 soft_max 概率即
            为边界标签的选择结果。
            if LAYER_NORM_USE:
                edus = self.edu_norm(edus)
        """
        super(Sim_Attn, self).__init__()
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.e_mlp = nn.Sequential(
            nn.Linear(encoder_size, hidden_size, bias=False),
            nn.SELU(),
            nn.Linear(hidden_size, K, bias=False),
            nn.SELU()
        ) if MLP_Layer == 2 else nn.Sequential(
            nn.Linear(encoder_size, K, bias=False),
            nn.SELU()
        )
        self.d_mlp = nn.Sequential(
            nn.Linear(decoder_size, hidden_size, bias=False),
            nn.SELU(),
            nn.Linear(hidden_size, K, bias=False),
            nn.SELU()
        ) if MLP_Layer == 2 else nn.Sequential(
            nn.Linear(encoder_size, K, bias=False),
            nn.SELU()
        )
        self.u1 = nn.Parameter(torch.empty(K, K, dtype=torch.float))
        self.u2 = nn.Parameter(torch.empty(K, num_labels, dtype=torch.float))
        # self.u3 = nn.Parameter(torch.empty(K, num_labels, dtype=torch.float))
        self.b = nn.Parameter(torch.empty(1, 1, num_labels, dtype=torch.float))
        nn.init.xavier_normal_(self.u1)
        nn.init.xavier_normal_(self.u2)
        # nn.init.xavier_normal_(self.u3)
        nn.init.xavier_normal_(self.b)

    def forward(self, e_outputs, d_outputs):
        """ :param e_outputs: (batch, length_encoder, encoder_size)
            :param d_outputs: (batch, length_decoder, decoder_size)
            encoder_size == decoder_size = HIDDEN_SIZE
        """
        h_e = self.e_mlp(e_outputs)  # (batch, length_encoder, K)
        h_d = self.d_mlp(d_outputs)  # (batch, K, length_decoder)
        part1 = h_e.matmul(self.u1)  # (batch, length_encoder, K)
        part1 = part1.bmm(h_d.transpose(1, 2)).transpose(1, 2).unsqueeze(-1)  # (batch, length_d, length_e, num_labels)
        part2 = h_e.matmul(self.u2).unsqueeze(1)  # (batch, 1<per decoder>, length_encoder, num_labels)
        # part3 = h_d.matmul(self.u3).unsqueeze(2)
        s = part1 + part2 + self.b  # [batch, length_decoder, length_encoder, num_labels]
        return s
