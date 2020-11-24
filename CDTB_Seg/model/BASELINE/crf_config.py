# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""


START_TAG, STOP_TAG = "<START>", "<STOP>"
tag2ids_ = {"O": 0, "B": 1, START_TAG: 2, STOP_TAG: 3}
ids2tag_ = {0: "O", 1: "B", 2: START_TAG, 3: STOP_TAG}
TAG_LABELS_ = ["O", "B", START_TAG, STOP_TAG]
