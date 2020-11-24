# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
from model.BASELINE.trainer import main as main_baseline
from model.GCN_basic.trainer import main as main_gcn
from model.ENC_DEC_GCN.trainer import main as main_enc_dec
from model.GCN_CRF.trainer import main as main_gcn_crf
from config import VERSION
from model.SEG_BOT.trainer import main as seg_bot_main
from model.ENC_DEC_GCN_ST_O.trainer import main as main_stronger
from model.Best_EDG.trainer import main as main_best

if __name__ == "__main__":
    if VERSION == 0:
        main_baseline()
    if VERSION == 1:
        main_gcn()
    if VERSION == 2 or VERSION == 11:
        main_enc_dec()
    if VERSION == 3:
        main_gcn_crf()
    if VERSION == 12:
        seg_bot_main()
    if VERSION == 10:
        main_stronger()
    if VERSION == 13:
        main_best()
