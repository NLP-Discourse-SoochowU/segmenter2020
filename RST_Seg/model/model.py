# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:  对原文件的修改，做解码的全进行和部分挑选策略。
"""
import torch
import torch.nn as nn
from config import *
from model.sim_attn import Sim_Attn
import torch.nn.functional as func
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
torch.manual_seed(SEED)


class BiGRUEDUEncoder(nn.Module):
    def __init__(self, input_size):
        super(BiGRUEDUEncoder, self).__init__()
        self.input_size = input_size
        self.masked_gru = MaskedGRU(input_size, HIDDEN_SIZE//2, bidirectional=True)
        self.token_scorer = nn.Linear(HIDDEN_SIZE, 1)
        self.drop = nn.Dropout(DROP_OUT)

    def forward(self, inputs, masks):
        """ inputs:  (batch * edu_num, word_len, emb)
            masks:  (batch * edu_num, word_len)
        """
        lengths = masks.sum(-1)
        outputs, hidden = self.masked_gru(inputs, lengths)
        token_score = self.token_scorer(outputs).squeeze(-1)
        token_score[masks == 0] = -1e8
        token_score = token_score.softmax(dim=-1) * masks.float()
        weighted_sum = (outputs * token_score.unsqueeze(-1)).sum(-2)
        outputs = hidden + self.drop(weighted_sum)  # mixed
        return outputs


class Segment_Model(nn.Module):
    def __init__(self, word_emb):
        super(Segment_Model, self).__init__()
        if not USE_ELMo and not USE_BERT:
            self.word_emb = nn.Embedding(word_emb.shape[0], WORDEMB_SIZE)
            self.word_emb.weight.data.copy_(torch.from_numpy(word_emb))
            self.word_emb.weight.requires_grad = True if EMBED_LEARN else False
        if USE_POS:
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
        self.encoder = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE // 2, bidirectional=True, batch_first=True)
        self.decoder = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE, batch_first=True)
        self.context_dense = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.bi_affine = nn.ModuleList(
            [Sim_Attn(HIDDEN_SIZE, HIDDEN_SIZE, 1, HIDDEN_SIZE) for _ in range(Head_NUM)])
        self.residualDrop = nn.Dropout(RESIDUAL_DROPOUT)
        self.token_scorer = nn.Linear(HIDDEN_SIZE, 1)

    def gcn_encode(self, word_ids, word_elmo_embeddings, word_bert_embeddings, pos_ids, masks, graph):
        """ 给定序列的编码部分
        """
        word_emb = word_bert_embeddings if USE_BERT else (word_elmo_embeddings if USE_ELMo else self.word_emb(word_ids))
        pos_emb = self.pos_emb(pos_ids) if USE_POS else None
        # (batch_size, padding_length, embedding_size)
        rnn_inputs = torch.cat([word_emb, pos_emb], dim=-1) if USE_POS else word_emb
        if masks is not None:
            lengths = masks.sum(-1)
            rnn_inputs = rnn_inputs * masks.unsqueeze(-1).float()
            rnn_inputs_packed = pack_padded_sequence(rnn_inputs, lengths, batch_first=True)
            rnn_outputs_packed, _ = self.sent_encode(rnn_inputs_packed)
            rnn_outputs, _ = pad_packed_sequence(rnn_outputs_packed, batch_first=True)
            input(rnn_outputs.size())
        else:
            rnn_outputs, _ = self.sent_encode(rnn_inputs)
        # token scorer
        if TOKEN_SCORE:
            token_score = self.token_scorer(rnn_outputs).squeeze(-1)
            token_score[masks == 0] = -1e8
            token_score = token_score.softmax(dim=-1) * masks.float()
            weighted_outputs = rnn_outputs * token_score.unsqueeze(-1)
            rnn_outputs = rnn_outputs + self.residualDrop(weighted_outputs)  # mixed
        last_gcn_outputs = rnn_outputs  # (batch, sent_len, hidden)
        count_ = 0
        for gcn in self.gcn_s:
            gcn_outputs = gcn(graph, last_gcn_outputs)
            count_ += 1
            if count_ < GCN_LAYER:
                last_gcn_outputs = last_gcn_outputs + self.residualDrop(gcn_outputs)
        # encoder
        e_out, hidden = self.encoder(last_gcn_outputs)
        # e_out = gcn_outputs + self.residualDrop(e_out)
        return e_out, hidden, last_gcn_outputs

    def bia_decode(self, e_out, hidden, gcn_outputs, decode_indices, decode_mask):
        batch_size = hidden.size(1)
        init_states = hidden.transpose(0, 1).contiguous().view(batch_size, -1)
        # print(e_out.size(), hidden.size(), gcn_outputs.size(), decode_indices.size(), decode_mask.size())
        # print(decode_indices, decode_indices.size())
        if Partial_IN:
            d_inputs = gcn_outputs[torch.arange(batch_size), decode_indices.transpose(0, 1)].transpose(0, 1)
            d_out, _ = self.decoder(d_inputs, init_states.unsqueeze(0))  # d_out = {batch, num_boundary, hidden}
        else:
            # 对 gcn 所有输出放入解码器
            # print(gcn_outputs.size(), init_states.unsqueeze(0).size())
            d_out, _ = self.decoder(gcn_outputs, init_states.unsqueeze(0))
            # print(d_out.size())
            # 抽取边界点之后的解码输出作为边界挑选的监测信息
            # decode_indices = decode_indices.unsqueeze(0).unsqueeze(0)
            # print(decode_indices.size())
            # print(d_out.size(), decode_indices.size())
            d_out = d_out[torch.arange(batch_size), decode_indices.transpose(0, 1)].transpose(0, 1)

        # 对挑选的解码输出进行边界挑选, 多头 bi_affine attention 进行边界选择的多角度控制
        bi_affine_attn = None
        for sim_attn in self.bi_affine:
            tmp_attn = sim_attn(e_out, d_out).squeeze(-1).unsqueeze(0)
            bi_affine_attn = tmp_attn if bi_affine_attn is None else torch.cat((bi_affine_attn, tmp_attn), dim=0)
        attn = torch.mean(bi_affine_attn, 0)
        # print(attn.size())

        # 解码输出边界概率
        decode_mask = decode_mask.float()
        mask_pad_ = (1 - decode_mask) * SMOO_VAL
        masked_bi_affine_attn = attn.mul(decode_mask) + mask_pad_
        boundary_predict = masked_bi_affine_attn.log_softmax(dim=2)
        return boundary_predict, bi_affine_attn

    def forward(self, inputs):
        word_ids, word_elmo_embed, word_bert_embed, pos_ids, graph, decode_indices, decode_mask, masks = inputs
        self.sent_encode.flatten_parameters()
        e_out, hidden, gcn_outputs = self.gcn_encode(word_ids, word_elmo_embed, word_bert_embed, pos_ids, masks, graph)
        tag_score, bi_affine_attn = self.bia_decode(e_out, hidden, gcn_outputs, decode_indices, decode_mask)
        return tag_score, bi_affine_attn

    def predict_(self, word_ids, word_elmo_embeddings, word_bert_embeddings, pos_ids, graph, masks=None):
        """ Evaluation preparation.
        """
        seq_len = word_ids.size()[1]
        e_out, hidden, gcn_outputs = self.gcn_encode(word_ids, word_elmo_embeddings, word_bert_embeddings, pos_ids,
                                                     masks, graph)
        # 以编码结尾作为初始状态对 gcn 结果整体编码
        state = hidden.transpose(0, 1).view(1, 1, -1)
        d_outs, _ = self.decoder(gcn_outputs, state)
        # 逐个解码边界
        start_idx, d_end, d_outputs = 0, False, None
        while not d_end:
            # 挑选输出信息
            d_out = d_outs[:, start_idx, :].unsqueeze(1)

            # 对输出和编码结果进行多头 bi_affine attention
            bi_affine_attn = None
            for sim_attn in self.bi_affine:
                tmp_attn = sim_attn(e_out, d_out).squeeze(-1)
                bi_affine_attn = tmp_attn if bi_affine_attn is None else torch.cat((bi_affine_attn, tmp_attn), dim=1)
            attn = torch.mean(bi_affine_attn, 1).unsqueeze(0)
            # 进行边界点选择，从左到右
            boundary_idx, start_idx, _ = self.select_boundary(attn, start_idx, seq_len)
            d_outputs = boundary_idx if d_outputs is None else torch.cat((d_outputs, boundary_idx), dim=0)
            if start_idx == seq_len:
                d_end = True
        return d_outputs

    @staticmethod
    def select_boundary(attn, state_idx, seq_len):
        """ attn: [batch, length_decoder, length_encoder]
                  (1, 1, n, 1)
            state_idx: tmp_area start idx
        """
        decode_mask = [0 for _ in range(state_idx)]
        decode_mask = decode_mask + [1 for _ in range(state_idx, seq_len)]
        decode_mask = torch.Tensor(decode_mask).float().cuda(CUDA_ID)
        mask_pad_ = (1 - decode_mask) * SMOO_VAL
        masked_attn = attn.mul(decode_mask) + mask_pad_  # make it small enough
        boundary_predict = torch.argmax(masked_attn.log_softmax(dim=-1)).unsqueeze(0)
        state_idx = (boundary_predict + 1).item()
        return boundary_predict, state_idx, masked_attn

    def fetch_attn(self, word_ids, pos_ids, graph, masks=None):
        """ Paper data preparation.
            解码方式特殊，编码方式一致，解码需要对每个序列的每个边界点处理出来，做顺序解码。
            注意，这款解码器包含了每句话的结束点作为标签之一，同样我们在target中也设置了这个边界进行学习。
        """
        word_emb = self.word_emb(word_ids)
        seq_len = word_emb.size()[1]
        pos_emb = self.pos_emb(pos_ids) if USE_POS else None
        rnn_inputs = torch.cat([word_emb, pos_emb], dim=-1) if USE_POS else word_emb
        if masks is not None:
            lengths = masks.sum(-1)
            rnn_inputs = rnn_inputs * masks.unsqueeze(-1).float()
            rnn_inputs_packed = pack_padded_sequence(rnn_inputs, lengths, batch_first=True)
            rnn_outputs_packed, _ = self.sent_encode(rnn_inputs_packed)
            rnn_outputs, _ = pad_packed_sequence(rnn_outputs_packed, batch_first=True)
        else:
            rnn_outputs, _ = self.sent_encode(rnn_inputs)
        count_ = 0
        gcn_outputs = rnn_outputs
        for gcn in self.gcn_s:
            gcn_outputs = gcn(graph, gcn_outputs)
            count_ += 1
            if count_ < GCN_LAYER:
                gcn_outputs = self.residualDrop(gcn_outputs)
        e_out, hidden = self.encoder(gcn_outputs)

        # 以编码结尾作为初始状态对gcn结果整体编码
        state = hidden.transpose(0, 1).view(1, 1, -1)
        d_outs, _ = self.decoder(gcn_outputs, state)

        # 逐个解码边界
        start_idx, d_end, d_outputs, attn_outputs = 0, False, None, None
        while not d_end:
            # 挑选输出信息
            d_out = d_outs[:, start_idx, :].unsqueeze(1)

            # 对输出和编码结果进行多头 bi_affine attention
            bi_affine_attn = None
            for sim_attn in self.bi_affine:
                tmp_attn = sim_attn(e_out, d_out).squeeze(-1)
                bi_affine_attn = tmp_attn if bi_affine_attn is None else torch.cat((bi_affine_attn, tmp_attn), dim=0)
            attn = torch.mean(bi_affine_attn, 0).unsqueeze(0)

            # 进行边界点选择，从左到右
            _, start_idx, prop_out = self.select_boundary(attn, start_idx, seq_len)
            attn_outputs = prop_out if attn_outputs is None else torch.cat((attn_outputs, prop_out), dim=1)
            if start_idx == seq_len:
                d_end = True
        return attn_outputs


class GCN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.empty(SYN_SIZE, input_size, hidden_size, dtype=torch.float))
        if Bias:
            self.bias = True
            self.b = nn.Parameter(torch.empty(SYN_SIZE, hidden_size, dtype=torch.float))
            self.sim_bias = nn.Parameter(torch.empty(1, 1, hidden_size, dtype=torch.float))
            nn.init.xavier_normal_(self.b)
            nn.init.xavier_normal_(self.sim_bias)
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
        self.nnDrop = nn.Dropout(RESIDUAL_DROPOUT)

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
            if SIM_BIAS:
                bias = self.sim_bias
            else:
                bias = (graph.float().view(batch_size * seq_len * seq_len, SYN_SIZE) @
                        self.b).view(batch_size, seq_len, seq_len, HIDDEN_SIZE)
                bias = bias.sum(2)
            h = h + bias
        return h

    def forward(self, graph, nodes):
        """ graph (bat, n, n, l) nodes (bat, n, h) transform into (bat, n, n, h)
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
