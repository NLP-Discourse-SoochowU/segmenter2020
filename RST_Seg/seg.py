# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description: Load pre_trained model and do segmentation.
"""
import numpy as np
import torch
from config import *
from config_path import *
from utils.file_util import *
model_path = "data/seg/seg.model"
seg_path_g = "data/seg/gold.tsv"
seg_path_p = "data/seg/pred.tsv"
seg_path_attn = "data/seg/attn.tsv"


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


def do_seg(word2ids, pos2ids, syn2ids):
    """ 分割测试集
    """
    _, test_set_raw = load_data(DATA_SETS_RAW)
    model = torch.load(model_path)
    model.eval()
    sent_g_list, sent_p_list = [], []
    for sentence in test_set_raw:
        tmp_gold_sent = "<PAD> "
        edus = sentence.edus
        sent_words_ids, sent_poses_ids, sent_tags_ids, graph_ids = [], [], [], []
        for i, edu in enumerate(edus):
            words = [word2ids[word] for word in edu.words]
            tmp_gold_sent += (" ".join(edu.words) + " <B> ")
            poses = [pos2ids[pos] for pos in edu.pos_tags]
            tags = ['O'] * (len(words) - 1)
            tags += ['B'] if i < len(edus) - 1 else ['O']
            tags = [tag2ids[tag] for tag in tags]
            sent_words_ids.extend(words)
            sent_poses_ids.extend(poses)
            sent_tags_ids.extend(tags)
        # root PAD
        sent_words_ids.insert(0, PAD_ID)
        sent_poses_ids.insert(0, PAD_ID)
        sent_tags_ids.insert(0, 0)  # 放置 0 将 root作为内部节点看待
        # (type, "head", "dep")
        for i, dep_pair in enumerate(sentence.dependency):
            if USE_ALL_SYN_INFO:
                graph_ids.append((i, i, syn2ids["self"]))
                graph_ids.append((dep_pair[1], dep_pair[2], syn2ids[dep_pair[0] + "-head"]))
                graph_ids.append((dep_pair[2], dep_pair[1], syn2ids[dep_pair[0] + "-dep"]))
            else:
                graph_ids.append((i, i, sync2ids["self"]))
                graph_ids.append((dep_pair[1], dep_pair[2], sync2ids["head"]))
                graph_ids.append((dep_pair[2], dep_pair[1], sync2ids["dep"]))
        # segment
        data_ite = (sent_words_ids, sent_poses_ids, graph_ids, sent_tags_ids)
        max_seq_len = len(sent_words_ids)
        if max_seq_len >= MAX_SEQ_LEN:
            continue  # GPU limitation
        word_ids, pos_ids, graph, _, _, masks = data_ids_prep(max_seq_len, data_ite)
        pred = model.decode_all(word_ids, pos_ids, graph, masks)
        predict = pred.data.cpu().numpy().tolist()
        # 句子，标准边界，预测边界
        sent_g_list.append(tmp_gold_sent.strip())
        tmp_pred_sent, sent_words = [], []
        edus = sentence.edus
        for i, edu in enumerate(edus):
            sent_words += edu.words
        sent_words.insert(0, "<PAD>")
        tmp_boundary = predict.pop(0)
        for idx, tok in enumerate(sent_words):
            tmp_pred_sent.append(tok)
            if idx == tmp_boundary:
                tmp_pred_sent.append("<B>")
                if len(predict) > 0:
                    tmp_boundary = predict.pop(0)
        tmp_pred_sent = " ".join(tmp_pred_sent)
        sent_p_list.append(tmp_pred_sent)
    write_iterate(sent_g_list, seg_path_g)
    write_iterate(sent_p_list, seg_path_p)


def model_attn_save(word2ids, pos2ids, syn2ids):
    _, test_set_raw = load_data(DATA_SETS_RAW)
    model = torch.load(model_path)
    model.eval()
    sent_attn_list = []
    sent_line_id = 1
    for sentence in test_set_raw:
        edus = sentence.edus
        sent_words_ids, sent_poses_ids, sent_tags_ids, graph_ids = [], [], [], []
        for i, edu in enumerate(edus):
            words = [word2ids[word] for word in edu.words]
            poses = [pos2ids[pos] for pos in edu.pos_tags]
            tags = ['O'] * (len(words) - 1)
            tags += ['B'] if i < len(edus) - 1 else ['O']
            tags = [tag2ids[tag] for tag in tags]
            sent_words_ids.extend(words)
            sent_poses_ids.extend(poses)
            sent_tags_ids.extend(tags)
        # root PAD
        sent_words_ids.insert(0, PAD_ID)
        sent_poses_ids.insert(0, PAD_ID)
        sent_tags_ids.insert(0, 0)  # 放置 0 将 root作为内部节点看待
        # (type, "head", "dep")
        for i, dep_pair in enumerate(sentence.dependency):
            if USE_ALL_SYN_INFO:
                graph_ids.append((i, i, syn2ids["self"]))
                graph_ids.append((dep_pair[1], dep_pair[2], syn2ids[dep_pair[0] + "-head"]))
                graph_ids.append((dep_pair[2], dep_pair[1], syn2ids[dep_pair[0] + "-dep"]))
            else:
                graph_ids.append((i, i, sync2ids["self"]))
                graph_ids.append((dep_pair[1], dep_pair[2], sync2ids["head"]))
                graph_ids.append((dep_pair[2], dep_pair[1], sync2ids["dep"]))
        # segment
        data_ite = (sent_words_ids, sent_poses_ids, graph_ids, sent_tags_ids)
        max_seq_len = len(sent_words_ids)
        if max_seq_len >= MAX_SEQ_LEN:
            continue
        word_ids, pos_ids, graph, _, _, masks = data_ids_prep(max_seq_len, data_ite)

        # key
        attn_outputs = model.fetch_attn(word_ids, pos_ids, graph, masks)
        attn_outputs = attn_outputs.data.cpu().numpy().tolist()[0]
        attn_str = "Line number: " + str(sent_line_id) + "\n[\n"
        for item in attn_outputs:
            for attn in item[1:]:
                attn_str += (str(attn) + " ")
            attn_str += "\n"
        attn_str += "]\n=======\n"
        sent_attn_list.append(attn_str)
        write_iterate(sent_attn_list, seg_path_attn)
        sent_line_id += 1


if __name__ == "__main__":
    word2ids_, pos2ids_, syn2ids_ = load_data(WORD2IDS), load_data(POS2IDS), load_data(SYN2IDS)
    # do_seg(word2ids_, pos2ids_, syn2ids_)
    model_attn_save(word2ids_, pos2ids_, syn2ids_)
