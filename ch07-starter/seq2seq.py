# coding: utf-8
import sys

sys.path.append("..")
from common_starter.time_layers import TimeEmbedding, TimeLSTM
from common_starter.base_model import BaseModel
import numpy as np


class Encoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype("f")
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype("f")
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b = np.zeros(4 * H).astype("f")

        self.embed = TimeLSTM(embed_W)

