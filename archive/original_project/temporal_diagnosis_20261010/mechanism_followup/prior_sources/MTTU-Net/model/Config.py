# -*- coding: utf-8 -*-
# @Author  : Zicheng Feng
# @File    : Config.py
# @Software: PyCharm
# coding=utf-8
import os
import torch
import time
import ml_collections

##########################################################################
# MTTU_Net configs
##########################################################################


def get_MTTU_Net_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.QKV_size = 128 * 4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 32  # base channel of U-Net
    config.transformer_channel = 128
    config.n_classes = 1

    # ********** useless **********
    # config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    return config
