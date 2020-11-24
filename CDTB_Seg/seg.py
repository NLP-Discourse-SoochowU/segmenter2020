# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description: Load pre_trained model and do segmentation.
"""
import numpy as np
import torch
from config import *
from path_config import *
from utils.file_util import *
model_path = "data/seg/seg.model"
seg_path_g = "data/seg/gold.tsv"
seg_path_p = "data/seg/pred.tsv"
seg_path_attn = "data/seg/attn.tsv"
USE_ENC_DEC = True


def data_ids_prep(max_seq_len, batch):
    """ Transform all the data into the form of ids.
    """
    word_ids, pos_ids, graph_ids, tag_ids = batch
    targets = None
    word_inputs = np.zeros([BATCH_SIZE, max_seq_len], dtype=np.long)
    pos_inputs = np.zeros([BATCH_SIZE, max_seq_len], dtype=np.long)
    graph_inputs = np.zeros([BATCH_SIZE, max_seq_len, max_seq_len, SYN_SIZE], np.uint8)
    tag_outputs = np.zeros([BATCH_SIZE, max_seq_len], dtype=np.long)  # 每一句对应的分子句的标签 0 1 1 1 2 1 1 2 ...
    masks = np.zeros([BATCH_SIZE, max_seq_len], dtype=np.uint8)
    seq_len = len(word_ids)
    word_inputs[0][:seq_len] = word_ids
    pos_inputs[0][:seq_len] = pos_ids
    tag_outputs[0][:seq_len] = tag_ids
    for x, y, z in graph_ids:
        # Use one-hot vector to represent the connection between nodes, 0 denotes no, 1 refers to yes.
        graph_inputs[0, x, y, z] = 1
    masks[0][:seq_len] = 1
    if USE_ENC_DEC:
        targets = np.where(tag_outputs[0] == 1)[0]
        if LearnFromEnd:
            targets = np.insert(targets, targets.shape[0], max_seq_len-1)
    # numpy2torch
    word_inputs = torch.from_numpy(word_inputs).long()
    pos_inputs = torch.from_numpy(pos_inputs).long()
    tag_outputs = torch.from_numpy(tag_outputs).long()
    graph_inputs = torch.from_numpy(graph_inputs).byte()
    targets = torch.from_numpy(targets).long()
    masks = torch.from_numpy(masks).byte()
    if USE_GPU:
        word_inputs = word_inputs.cuda(CUDA_ID)
        pos_inputs = pos_inputs.cuda(CUDA_ID)
        tag_outputs = tag_outputs.cuda(CUDA_ID)
        graph_inputs = graph_inputs.cuda(CUDA_ID)
        targets = targets.cuda(CUDA_ID)
        masks = masks.cuda(CUDA_ID)
    return word_inputs, pos_inputs, graph_inputs, tag_outputs, targets, masks


def do_seg(word2ids, pos2ids):
    """ 分割测试集
    """
    model = torch.load(model_path)
    model.eval()
    _, _, test_set_raw = load_data(RAW_DATA_SETS)
    instances, tags = test_set_raw
    sent_g_list, sent_p_list = [], []
    for instance, tag in zip(instances, tags):
        sent_words, sent_poses, graph = instance
        tag_seq = tag
        sent_words_ids = [word2ids[word] for word in sent_words]
        sent_poses_ids = [pos2ids[pos] for pos in sent_poses]
        sent_tags_ids = [tag2ids[tag_] for tag_ in tag_seq]
        # 句法信息: 目前只考虑方向
        graph_ids = []
        for graph_edge in graph:
            from_, to_, syn_ = graph_edge
            graph_ids.append((from_, to_, sync2ids[syn_]))
        # segment
        data_ite = (sent_words_ids, sent_poses_ids, graph_ids, sent_tags_ids)
        max_seq_len = len(sent_words_ids)
        if max_seq_len >= MAX_SEQ_LEN:
            continue  # GPU limitation
        word_ids, pos_ids, graph, _, _, masks = data_ids_prep(max_seq_len, data_ite)
        inputs = (word_ids, pos_ids, graph, None, None, masks)
        pred = model(inputs)
        # pred = model.decode_all(word_ids, pos_ids, graph, masks)
        # predict
        predict = pred.data.cpu().numpy().tolist()
        tmp_pred_sent = []
        tmp_boundary = predict.pop(0)
        for idx, tok in enumerate(sent_words):
            tmp_pred_sent.append(tok)
            if idx == tmp_boundary:
                tmp_pred_sent.append("<B>")
                if len(predict) > 0:
                    tmp_boundary = predict.pop(0)
        tmp_pred_sent = " ".join(tmp_pred_sent)
        sent_p_list.append(tmp_pred_sent)
        # print(tmp_pred_sent)
        # gold
        sent_tags_ids = np.array(sent_tags_ids)
        # print(sent_tags_ids)
        # input(np.where(sent_tags_ids == 1)[0])
        targets = np.where(sent_tags_ids == 1)[0].tolist()
        targets.append(max_seq_len - 1)
        # input(targets)
        tmp_gold_sent = []
        tmp_boundary = targets.pop(0)
        for idx, tok in enumerate(sent_words):
            tmp_gold_sent.append(tok)
            if idx == tmp_boundary:
                tmp_gold_sent.append("<B>")
                if len(targets) > 0:
                    tmp_boundary = targets.pop(0)
        tmp_gold_sent = " ".join(tmp_gold_sent)
        sent_g_list.append(tmp_gold_sent)
        # input(tmp_gold_sent)
    write_iterate(sent_p_list, seg_path_p)
    write_iterate(sent_g_list, seg_path_g)


def model_attn_save(word2ids, pos2ids):
    """ 分割测试集
    """
    model = torch.load(model_path)
    model.eval()
    _, _, test_set_raw = load_data(RAW_DATA_SETS)
    instances, tags = test_set_raw
    sent_attn_list = []
    sent_line_id = 1
    for instance, tag in zip(instances, tags):
        sent_words, sent_poses, graph = instance
        tag_seq = tag
        sent_words_ids = [word2ids[word] for word in sent_words]
        sent_poses_ids = [pos2ids[pos] for pos in sent_poses]
        sent_tags_ids = [tag2ids[tag_] for tag_ in tag_seq]
        # 句法信息: 目前只考虑方向
        graph_ids = []
        for graph_edge in graph:
            from_, to_, syn_ = graph_edge
            graph_ids.append((from_, to_, sync2ids[syn_]))
        # segment
        data_ite = (sent_words_ids, sent_poses_ids, graph_ids, sent_tags_ids)
        max_seq_len = len(sent_words_ids)
        if max_seq_len >= MAX_SEQ_LEN:
            continue  # GPU limitation
        word_ids, pos_ids, graph, _, _, masks = data_ids_prep(max_seq_len, data_ite)
        # inputs = (word_ids, pos_ids, graph, None, None, masks)
        # pred = model(inputs)

        attn_outputs = model.fetch_attn(word_ids, pos_ids, graph, masks)
        attn_outputs = attn_outputs.data.cpu().numpy().tolist()[0]
        attn_str = "Line number: " + str(sent_line_id) + "\n[\n"
        for item in attn_outputs:
            for attn in item:
                attn_str += (str(attn) + " ")
            attn_str += "\n"
        attn_str += "]\n=======\n"
        sent_attn_list.append(attn_str)
        write_iterate(sent_attn_list, seg_path_attn)
        sent_line_id += 1


def obtain_root_b():
    """ 获取指向 root 节点的边界的个数 // 所有边界的个数
        统计, 位置2指向的位置1节点为root节点或者0，并且句法关系为0的个数 / 所有边个数
        1. 找到 root 节点；
        2. 找边界列表；统计边界个数；
        3. 统计边界指向root节点的元素个数；
    """
    train, dev, test = load_data(DATA_SETS)
    sets = train + dev + test
    total_b, root_b = 0., 0.
    for instance in sets:
        _, _, graph_ids, sent_tags_ids = instance
        # 1.
        root_idx = graph_ids[1][1]
        # 2.
        boundaries = []
        for idx, bound in enumerate(sent_tags_ids):
            if bound == 1:
                boundaries.append(idx)
        boundaries.append(len(sent_tags_ids) - 1)
        total_b += len(boundaries)
        # 3. 存在即为真理
        for idx in range(2, len(graph_ids)):
            tmp_node = graph_ids[idx]
            if tmp_node[1] in boundaries and tmp_node[0] == root_idx:
                root_b += 1
    print(root_b / total_b)


if __name__ == "__main__":
    # word2ids_, pos2ids_ = load_data(WORD2IDS), load_data(POS2IDS)
    # do_seg(word2ids_, pos2ids_)
    # model_attn_save(word2ids_, pos2ids_)
    obtain_root_b()
