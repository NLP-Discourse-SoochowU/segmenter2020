# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
import torch
import torch.nn as nn
from config import *
import torch.nn.functional as func


class GCN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 参数过多 over-parameters
        self.W = nn.Parameter(torch.empty(SYN_SIZE, input_size, hidden_size, dtype=torch.float))
        if Bias:
            self.bias = True
            self.b = nn.Parameter(torch.empty(SYN_SIZE, hidden_size, dtype=torch.float))
            nn.init.xavier_normal_(self.b)
        nn.init.xavier_normal_(self.W)
        # part 2
        self.label_rank = nn.Parameter(torch.empty(SYN_SIZE, R, dtype=torch.float))
        self.label_Q = nn.Parameter(torch.empty(input_size, R, dtype=torch.float))
        self.label_P = nn.Parameter(torch.empty(R, hidden_size, dtype=torch.float))
        self.label_b = nn.Parameter(torch.empty(R, hidden_size, dtype=torch.float))
        nn.init.xavier_normal_(self.label_rank)
        nn.init.xavier_normal_(self.label_Q)
        nn.init.xavier_normal_(self.label_P)
        nn.init.xavier_normal_(self.label_b)

    def batch_rank_optimized_gcn_h(self, g, x):
        """ x refers to nodes.
            a piece of data (a single sequence, batch_size 需设置为 1):
                g = n * n * l
                x = n * h1
                L = l * r
                Q = h1 * r
                P = r * h2
                b = r * h2  (2*h2+h1+l)*r， h1 = h * l --> (2*h2 + (h + 1) * l) * r
                            --> 2 * h2 * l + (h+1) * l * r == k*l*h
            process:
                x.Q.diagonalized(g.L).P + g.L.b

            original:
                l * h * h2  == l*h^2
            process:
                W.X + b
            input_ demotes the ranked x (n, n, r) and we aim to transform it into (n, n, r, r).
            given (n, r), diag_embed results in (n, r, r)
            suppose given (n, n, r) --> (n, n, r, r)
        """
        g = g.float()
        # (b, n, n, l) matmul (l, r) --> (b, n, n, r)
        ranked_g = g.matmul(self.label_rank)
        g_diagonalized = torch.diag_embed(ranked_g)  # (b, n, n, r, r)
        # (h1, r) matmul (b, n, n, r, r) --> (b, n, n, h1, r)
        w_tran = self.label_Q.matmul(g_diagonalized)
        # (b, n, n, h1, r) matmul (r, h2) --> (b, n, n, h1, h2)
        w_tran = w_tran.matmul(self.label_P)
        # (b, n, 1, 1, h1) matmul (b, n, n, h1, h2) --> (b, n, n, h2)
        x = x.unsqueeze(2)
        x = x.unsqueeze(2)
        part_a = x.matmul(w_tran).squeeze(3)
        # (b, n, n, r) matmul (r, h2) --> (b, n, n, h2)
        part_b = ranked_g.matmul(self.label_b)
        gc_x = part_a + part_b if Bias else part_a
        gc_x = torch.sum(gc_x, 2, False)
        return gc_x

    def basic_gcn_h(self, graph, nodes):
        """ Lstm represented nodes of a batch of sentences and a batch of graphs
            with syntactic information.
                nodes (bat, n, h), graph (bat, n, n, l)
                basic + improved
        """
        batch_size, seq_len, _ = nodes.size()
        # graph (bat, n * l, n)
        g = graph.transpose(2, 3).float().contiguous().view(batch_size, seq_len * SYN_SIZE, seq_len)
        # x: (bat, n, l * h)
        x = g.bmm(nodes).view(batch_size, seq_len, SYN_SIZE * HIDDEN_SIZE)
        # h: (bat, n, h)
        h = x.matmul(self.W.view(SYN_SIZE * HIDDEN_SIZE, HIDDEN_SIZE))
        if Bias:
            bias = (graph.float().view(batch_size * seq_len * seq_len, SYN_SIZE) @
                    self.b).view(batch_size, seq_len, seq_len, HIDDEN_SIZE)
            bias = bias.sum(2)
            h = h + bias
        return h

    def forward(self, graph, nodes):
        """ graph (bat, n, n, l)
            nodes (bat, n, h)
            transform into (bat, n, n, h)
        """
        batch_size, seq_len, _ = nodes.size()
        if BASELINE:
            hidden_rep = self.basic_gcn_h(graph, nodes)
        else:
            hidden_rep = self.batch_rank_optimized_gcn_h(graph, nodes)
        norm = graph.view(batch_size, seq_len, seq_len * SYN_SIZE).sum(-1).float().unsqueeze(-1) + 1e-10
        # h: (bat, n, h)
        hidden_rep = func.relu(hidden_rep / norm)
        return hidden_rep
