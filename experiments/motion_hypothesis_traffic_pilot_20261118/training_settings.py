"""Shared settings for paired training runs in this experiment."""
import random

import numpy as np
import torch


TRAIN_SEED = 46


def reset_training_rng():
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    torch.use_deterministic_algorithms(True)
    return np.random.default_rng(TRAIN_SEED)
