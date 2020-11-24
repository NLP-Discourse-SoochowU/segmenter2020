# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
# Experimental
VERSION, SET = 11, 95
USE_GPU, CUDA_ID = True, 1
BASELINE, Bias, Sim_Bias = True, False, False
GCN_LAYER, RNN_TYPE, RNN_LAYER = 1, "GRU", 3
USE_SIM_ATTN, K, Head_NUM = True, 256, 2
HIDDEN_SIZE, MLP_Layer = 384, 1
LR_DECAY, LR = False, 0.001
BATCH_SIZE, N_EPOCH, LOG_EVE, EVA_EVE = 1, 16, 20, 120
SAVE_MODEL = False

DROPOUT = 0.2
ENC_DEC_DROPOUT = 0
RESIDUAL_DROPOUT = 0.2

# statistic
R = 32
PAD = "<PAD>"
PAD_ID = 0
UNK = "<UNK>"
UNK_ID = 1
# 目测以 1 为批量，用不到 PAD，暂时将 PAD 以及对应 ID 去除
tag2ids = {"O": 0, "B": 1, PAD: 2}  # tag2ids = {"O": 0, "B": 1, PAD: 2, START_TAG: 3, STOP_TAG: 4}
ids2tag = {0: "O", 1: "B"}
sync2ids = {"head": 0, "dep": 1, "self": 2}
TAG_LABELS = ["O", "B"]
SEED = 7
USE_ALL_SYN_INFO = False
SYN_SIZE = 81 if USE_ALL_SYN_INFO else 3
POS_TAG_NUM, POS_TAG_SIZE = 36, 30
WORDEMB_SIZE = 300
EMBED_LEARN = False
L2 = 1e-5
MAX_SEQ_LEN = 140
SMOO_VAL = -1e2
PRINT_EVE = 1000
LearnFromEnd = True
VERSION2DESC = {0: "Bi_LSTM + CRF baseline.", 1: "GCN Basic", 2: "GCN enc + att dec", 3: "GCN + CRF", 11: "", 12: ""}
DEV_SIZE = 200
