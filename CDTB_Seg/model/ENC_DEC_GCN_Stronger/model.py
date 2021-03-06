# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import *


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
        nn.init.xavier_normal_(self.u1)
        nn.init.xavier_normal_(self.u2)

    def forward(self, e_outputs, d_outputs):
        """ :param e_outputs: (batch, length_encoder, encoder_size)
            :param d_outputs: (batch, length_decoder, decoder_size)
            encoder_size == decoder_size = HIDDEN_SIZE
        """
        h_e = self.e_mlp(e_outputs)  # (batch, length_encoder, K)
        h_d = self.d_mlp(d_outputs).transpose(1, 2)  # (batch, K, length_decoder)
        part1 = h_e.matmul(self.u1)  # (batch, length_encoder, K)
        part1 = part1.bmm(h_d).transpose(1, 2).unsqueeze(-1)  # (batch, length_decoder, length_encoder, num_labels)
        part2 = h_e.matmul(self.u2).unsqueeze(1)  # (batch, 1<per decoder>, length_encoder, num_labels)
        s = part1 + part2  # [batch, length_decoder, length_encoder, num_labels]
        return s


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
        # encoder + decoder
        self.encoder = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE // 2, bidirectional=True, batch_first=True)
        self.decoder = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE, batch_first=True)
        self.context_dense = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.bi_affine = nn.ModuleList([Sim_Attn(HIDDEN_SIZE, HIDDEN_SIZE, 1, HIDDEN_SIZE) for _ in range(Head_NUM)])
        self.nnDropout = nn.Dropout(ENC_DEC_DROPOUT)

    def enc_decode_(self, gcn_hidden, decode_mask, decode_indices):
        # 对 gcn 特征抽取之后的输出输入到编码器进行编码
        e_out, hidden = self.encoder(gcn_hidden)
        e_out = self.nnDropout(e_out)

        # 以编码器最后的隐层作为解码器的初始状态，以 gcn 输出作为解码器输入进行解码
        init_states = hidden.transpose(0, 1).view(1, -1)
        d_outs, _ = self.decoder(gcn_hidden, init_states.unsqueeze(0))

        # 挑选解码结果中初始节点以及各个边界点之后的解码结果作为边界挑选的监察点
        decode_indices = decode_indices.unsqueeze(0).unsqueeze(0)
        d_out = d_outs[torch.arange(BATCH_SIZE), decode_indices].squeeze(0)

        # 对挑选出来的节点和编码器结果做多头注意力，挑选边界点 (batch, length_decoder, length_encoder)
        bi_affine_attn = None
        for sim_attn in self.bi_affine:
            tmp_attn = sim_attn(e_out, d_out).squeeze(-1)
            bi_affine_attn = tmp_attn if bi_affine_attn is None else torch.cat((bi_affine_attn, tmp_attn), dim=0)
        attn = torch.mean(bi_affine_attn, 0).unsqueeze(0)  # 按照维度0进行取平均操作

        # 对各个区域的边界点筛选
        decode_mask = decode_mask.unsqueeze(0).float()
        mask_pad_ = (1 - decode_mask) * SMOO_VAL
        masked_attn = attn.mul(decode_mask) + mask_pad_
        boundary_predict = masked_attn.log_softmax(dim=2)
        return boundary_predict, bi_affine_attn

    @staticmethod
    def select_boundary(bi_affine_attn, state_idx, seq_len):
        """ attn: [batch, length_decoder, length_encoder, num_labels]
                  (1, 1, n, 1)
            state_idx: tmp_area start idx
        """
        decode_mask = [0 for _ in range(state_idx)]
        decode_mask = decode_mask + [1 for _ in range(state_idx, seq_len)]
        decode_mask = torch.Tensor(decode_mask).float().cuda(CUDA_ID)
        # decode_mask = decode_mask.unsqueeze(0).float()
        mask_pad_ = (1 - decode_mask) * SMOO_VAL
        masked_bi_affine_attn = bi_affine_attn.mul(decode_mask) + mask_pad_  # make it small enough
        # scoring
        boundary_predict = torch.argmax(masked_bi_affine_attn.log_softmax(dim=-1)).unsqueeze(0)
        state_idx = (boundary_predict + 1).item()
        return boundary_predict, state_idx

    def gen_loss(self, inputs=None, target=None):
        word_ids, pos_ids, graph, decode_indices, decode_mask, masks = inputs
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
        gcn_outputs = rnn_outputs
        for gcn in self.gcn_s:
            gcn_outputs = gcn(graph, gcn_outputs)
        tag_score, attn = self.enc_decode_(gcn_outputs, decode_mask, decode_indices)
        score_ = tag_score.squeeze(0)
        loss_pred = func.nll_loss(score_, target)
        # 对多头注意力进行两两相似性计算
        loss_difference = 0.
        for idx in range(Head_NUM):
            for c_idx in range(idx + 1, Head_NUM):
                tmp_loss = func.cosine_embedding_loss(attn[idx], attn[c_idx], target=torch.Tensor([-1]).cuda(CUDA_ID))
                loss_difference += tmp_loss
        loss_ = loss_pred + loss_difference
        return loss_

    def forward(self, inputs=None):
        """ 解码方式特殊，编码方式一致，解码需要对每个序列的每个边界点处理出来，做顺序解码。
            注意，这款解码器包含了每句话的结束点作为标签之一，同样我们在target中也设置了这个边界进行学习。
        """
        word_ids, pos_ids, graph, _, _, masks = inputs
        word_emb = self.word_emb(word_ids)
        seq_len = word_emb.size()[1]
        pos_emb = self.pos_emb(pos_ids)
        rnn_inputs = torch.cat([word_emb, pos_emb], dim=-1)
        lengths = masks.sum(-1)
        rnn_inputs = rnn_inputs * masks.unsqueeze(-1).float()
        rnn_inputs_packed = pack_padded_sequence(rnn_inputs, lengths, batch_first=True)
        rnn_outputs_packed, _ = self.sent_encode(rnn_inputs_packed)
        rnn_outputs, _ = pad_packed_sequence(rnn_outputs_packed, batch_first=True)
        # (batch, seq_len, hidden)
        gcn_outputs = rnn_outputs
        for gcn in self.gcn_s:
            gcn_outputs = gcn(graph, gcn_outputs)

        # 对 gcn 输出整体编码
        e_out, hidden = self.encoder(gcn_outputs)

        # 以编码结尾作为初始状态, 以 gcn 输出作为解码器整体解码
        state = hidden.transpose(0, 1).view(1, 1, -1)  # (batch, xxx, hidden)
        d_outs, _ = self.decoder(gcn_outputs, state)

        # 根据解码器输出，逐个挑选边界
        start_idx, d_end, d_outputs = 0, False, None
        while not d_end:
            # 当前位置解码器输出
            d_out = d_outs[:, start_idx, :].unsqueeze(1)

            # 多头注意力
            bi_affine_attn = None
            for sim_attn in self.bi_affine:
                tmp_attn = sim_attn(e_out, d_out).squeeze(-1)
                bi_affine_attn = tmp_attn if bi_affine_attn is None else torch.cat((bi_affine_attn, tmp_attn), dim=0)
            attn = torch.mean(bi_affine_attn, 0).unsqueeze(0)

            # 进行边界点位置选择，更新当前位置
            boundary_idx, start_idx = self.select_boundary(attn, start_idx, seq_len)
            d_outputs = boundary_idx if d_outputs is None else torch.cat((d_outputs, boundary_idx), dim=0)
            if start_idx == seq_len:
                d_end = True
        return d_outputs
