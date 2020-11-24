# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description: Load pre_trained model and do segmentation.
"""
import numpy as np
from config_path import DATA_SETS_RAW, WORD2IDS, POS2IDS, SYN2IDS
import torch
from config import *
from utils.file_util import *
if USE_ELMo:
    from allennlp.modules.elmo import batch_to_ids
    from allennlp.modules.elmo import Elmo
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/" \
                   "elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo" \
                  "_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    elmo = Elmo(options_file, weight_file, 2, dropout=0)
ELMO_ROOT_PAD = torch.zeros(1, 1024)

model_path = "data/seg.model"
RAW_SENT_PATH = "data/seg/TEST2SEG"
seg_path_g = "data/seg/gold.tsv"
seg_path_p = "data/seg/pred.tsv"
seg_path_attn = "data/seg/attn.tsv"


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
        sent_token_list = []
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
            sent_token_list += edu.words
        # root PAD
        sent_words_ids.insert(0, PAD_ID)
        sent_poses_ids.insert(0, PAD_ID)
        sent_tags_ids.insert(0, 0)  # 放置 0 将 root作为内部节点看待

        # token2elmo
        if USE_ELMo:
            sents_tokens_ids = batch_to_ids([sent_token_list])  # (1, sent_len)
            # 根据整句进行向量获取
            tmp_sent_tokens_emb = elmo(sents_tokens_ids)["elmo_representations"][0][0]
            tmp_sent_tokens_emb = torch.cat((ELMO_ROOT_PAD, tmp_sent_tokens_emb), 0)
        else:
            tmp_sent_tokens_emb = None
        # token2bert
        tmp_sent_tokens_bert = None

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
        data_ite = (sent_words_ids, sent_poses_ids, graph_ids, sent_tags_ids, tmp_sent_tokens_emb, tmp_sent_tokens_bert)
        max_seq_len = len(sent_words_ids)
        if max_seq_len >= MAX_SEQ_LEN:
            continue  # GPU limitation
        word_ids, word_elmo_embeddings, pos_ids, graph, _, _, masks = data_ids_prep(max_seq_len, data_ite)
        pred = model.predict_(word_ids, word_elmo_embeddings, None, pos_ids, graph, masks)
        predict = pred.data.cpu().numpy().tolist()
        print(predict, " \n ", tmp_gold_sent)
        input()
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


def data_ids_prep(max_seq_len, batch):
    """ Transform all the data into the form of ids.
    """
    word_ids, pos_ids, graph_ids, tag_ids, word_elmo, word_bert = batch
    targets = None
    word_inputs = np.zeros([BATCH_SIZE, max_seq_len], dtype=np.long)
    word_elmo_embeddings = np.zeros([BATCH_SIZE, max_seq_len, 1024], dtype=np.float)
    pos_inputs = np.zeros([BATCH_SIZE, max_seq_len], dtype=np.long)
    graph_inputs = np.zeros([BATCH_SIZE, max_seq_len, max_seq_len, SYN_SIZE], np.uint8)
    tag_outputs = np.zeros([BATCH_SIZE, max_seq_len], dtype=np.long)  # 每一句对应的分子句的标签 0 1 1 1 2 1 1 2 ...
    masks = np.zeros([BATCH_SIZE, max_seq_len], dtype=np.uint8)
    seq_len = len(word_ids)
    word_inputs[0][:seq_len] = word_ids
    word_elmo_embeddings[0][:seq_len][:] = word_elmo.detach().numpy()
    pos_inputs[0][:seq_len] = pos_ids
    tag_outputs[0][:seq_len] = tag_ids
    for x, y, z in graph_ids:
        # Use one-hot vector to represent the connection between nodes, 0 denotes no, 1 refers to yes.
        graph_inputs[0, x, y, z] = 1
    masks[0][:seq_len] = 1
    if USE_ENC_DEC:
        targets = np.where(tag_outputs[0] == 1)[0]
        targets = np.insert(targets, targets.shape[0], max_seq_len-1)
    # numpy2torch
    word_inputs = torch.from_numpy(word_inputs).long()
    word_elmo_embeddings = torch.from_numpy(word_elmo_embeddings).float()
    pos_inputs = torch.from_numpy(pos_inputs).long()
    tag_outputs = torch.from_numpy(tag_outputs).long()
    graph_inputs = torch.from_numpy(graph_inputs).byte()
    targets = torch.from_numpy(targets).long()
    masks = torch.from_numpy(masks).byte()
    if USE_GPU:
        word_inputs = word_inputs.cuda(CUDA_ID)
        word_elmo_embeddings = word_elmo_embeddings.cuda(CUDA_ID)
        pos_inputs = pos_inputs.cuda(CUDA_ID)
        tag_outputs = tag_outputs.cuda(CUDA_ID)
        graph_inputs = graph_inputs.cuda(CUDA_ID)
        targets = targets.cuda(CUDA_ID)
        masks = masks.cuda(CUDA_ID)
    return word_inputs, word_elmo_embeddings, pos_inputs, graph_inputs, tag_outputs, targets, masks


def assign_all():
    """ 对分割结果分配到 TEST 数据集中
    """
    sent2name = dict()
    # 构建第一句到文件名的映射
    for f_name in os.listdir(RAW_SENT_PATH):
        if f_name.endswith(".out"):
            f_path = os.path.join(RAW_SENT_PATH, f_name)
            f_obj = open(f_path, "r")
            sent = "".join(f_obj.readlines()[0].strip().lower().split())
            sent2name[sent] = f_name
    # 构建真实结果中的每句话的raw
    f_obj = open(seg_path_p, "r")
    sents_seg = f_obj.readlines()
    tmp_path = None
    for sent in sents_seg:
        raw_sent = sent.strip().lower()[5:]
        raw_sent = "".join(raw_sent.split("<b>")).strip()
        raw_sent = "".join(raw_sent.split())
        if raw_sent in sent2name.keys():
            tmp_fn = sent2name[raw_sent]
            tmp_path = os.path.join(RAW_SENT_PATH, tmp_fn + ".auto.edus")
        # 数据写入
        edus = sent.strip()[5:].split("<B>")
        for edu in edus:
            edu = edu.strip()
            if len(edu) == 0:
                continue
            write_append(edu, tmp_path)


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
        word_ids, word_elmo_embeddings, pos_ids, graph, _, _, masks = data_ids_prep(max_seq_len, data_ite)

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
    do_seg(word2ids_, pos2ids_, syn2ids_)
    # assign_all()
    # model_attn_save(word2ids_, pos2ids_, syn2ids_)
