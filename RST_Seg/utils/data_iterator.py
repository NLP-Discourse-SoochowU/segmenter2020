from config import *
import numpy as np
import torch


def data_ids_prep(num_batch, max_seq_len, max_dec_len, batch):
    """ Transform all the data into the form of ids.
    """
    word_inputs = np.zeros([num_batch, max_seq_len], dtype=np.long)
    word_elmo_embeddings = np.zeros([num_batch, max_seq_len, 1024], dtype=np.float)
    word_bert_embeddings = np.zeros([num_batch, max_seq_len, 768], dtype=np.float)
    pos_inputs = np.zeros([num_batch, max_seq_len], dtype=np.long)
    graph_inputs = np.zeros([num_batch, max_seq_len, max_seq_len, SYN_SIZE], np.uint8)
    tag_outputs = np.zeros([num_batch, max_seq_len], dtype=np.long)  # 每一句对应的分子句的标签 0 1 1 1 2 1 1 2 ...
    masks = np.zeros([num_batch, max_seq_len], dtype=np.uint8)
    decode_indices = np.zeros([num_batch, max_dec_len], dtype=np.long)
    decode_mask = np.zeros([num_batch, max_dec_len, max_seq_len], dtype=np.uint8)
    targets = np.zeros([num_batch, max_dec_len], dtype=np.long)
    for i, (word_ids, pos_ids, graph_ids, tag_ids, word_elmo, word_bert) in enumerate(batch):
        # input(tag_ids)
        seq_len = len(word_ids)
        word_inputs[i][:seq_len] = word_ids[:]
        if USE_ELMo:
            word_elmo_embeddings[i][:seq_len][:] = word_elmo.detach().numpy()
        if USE_BERT:
            word_bert_embeddings[i][:seq_len][:] = word_bert.detach().numpy()
        pos_inputs[i][:seq_len] = pos_ids[:]
        tag_outputs[i][:seq_len] = tag_ids[:]
        for x, y, z in graph_ids:
            # Use one-hot vector to represent the connection between nodes, 0 denotes no, 1 refers to yes.
            graph_inputs[i, x, y, z] = 1
        masks[i][:seq_len] = 1
        # print(tag_outputs[i])
        # print(np.where(tag_outputs[i] == 1))
        # print(np.where(tag_outputs[i] == 1)[0])
        decode_in = np.where(tag_outputs[i] == 1)[0] + 1
        # print(decode_in)
        decode_in = np.insert(decode_in, 0, 0)  # 将第一个元素插入，作为第一个解码的首位
        # print(decode_in)
        decode_in_len = decode_in.shape[0]
        decode_indices[i][:decode_in_len] = decode_in[:]
        for idx in range(decode_in_len):
            decode_idx_begin = decode_indices[i][idx]
            decode_mask[i][idx][decode_idx_begin:] = 1
        targets_ = np.where(tag_outputs[i] == 1)[0]
        targets_ = np.insert(targets_, targets_.shape[0], max_seq_len-1)  # 终点
        targets[i][:decode_in_len] = targets_[:]
    return word_inputs, word_elmo_embeddings, word_bert_embeddings, pos_inputs, graph_inputs, tag_outputs, \
        decode_indices, decode_mask, targets, masks


def gen_batch_iter(training_set, batch_s=BATCH_SIZE):
    """ a batch 2 numpy data.
    """
    random_instances = np.random.permutation(training_set)
    num_instances = len(training_set)
    offset = 0
    while offset < num_instances:
        batch = random_instances[offset: min(num_instances, offset + batch_s)]
        num_batch = batch.shape[0]
        lengths = np.zeros(num_batch, dtype=np.int)
        decode_nums = np.zeros(num_batch, dtype=np.int)
        for i, (word_ids, _, _, tag_ids, _, _) in enumerate(batch):
            lengths[i] = len(word_ids)
            decode_nums[i] = sum(tag_ids)
        sort_indices = np.argsort(-lengths)  # 从大到小排列，存放对应下标
        lengths = lengths[sort_indices]
        batch = batch[sort_indices]
        max_seq_len = lengths.max()  # 每个批次里面选择最大值进行 pad
        max_dec_len = decode_nums.max() + 1  # 增加一个首位
        if max_seq_len >= MAX_SEQ_LEN:
            # GPU limitation
            offset = offset + batch_s
            continue
        word_inputs, word_elmo_embeddings, word_bert_embeddings, pos_inputs, graph_inputs, tag_outputs, \
            decode_indices, decode_mask, targets, masks = data_ids_prep(num_batch, max_seq_len, max_dec_len, batch)
        offset = offset + batch_s
        # numpy2torch
        word_inputs = torch.from_numpy(word_inputs).long()
        if USE_ELMo:
            word_elmo_embeddings = torch.from_numpy(word_elmo_embeddings).float()
        if USE_BERT:
            word_bert_embeddings = torch.from_numpy(word_bert_embeddings).float()
        pos_inputs = torch.from_numpy(pos_inputs).long()
        tag_outputs = torch.from_numpy(tag_outputs).long()
        graph_inputs = torch.from_numpy(graph_inputs).byte()
        decode_indices = torch.from_numpy(decode_indices).long()
        decode_mask = torch.from_numpy(decode_mask).byte()
        targets = torch.from_numpy(targets).long()
        masks = torch.from_numpy(masks).byte()
        if USE_GPU:
            word_inputs = word_inputs.cuda(CUDA_ID)
            if USE_ELMo:
                word_elmo_embeddings = word_elmo_embeddings.cuda(CUDA_ID)
            if USE_BERT:
                word_bert_embeddings = word_bert_embeddings.cuda(CUDA_ID)
            pos_inputs = pos_inputs.cuda(CUDA_ID)
            tag_outputs = tag_outputs.cuda(CUDA_ID)
            graph_inputs = graph_inputs.cuda(CUDA_ID)
            decode_indices = decode_indices.cuda(CUDA_ID)
            decode_mask = decode_mask.cuda(CUDA_ID)
            targets = targets.cuda(CUDA_ID)
            masks = masks.cuda(CUDA_ID)
        yield (word_inputs, word_elmo_embeddings, word_bert_embeddings, pos_inputs, graph_inputs, decode_indices,
               decode_mask, masks), (tag_outputs, targets)
