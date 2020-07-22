# coding: utf-8
import sys

sys.path.append("..")
import numpy as np
from common.util import preprocess, create_co_matrix, ppmi, most_similar
from dataset import ptb
from sklearn.utils.extmath import randomized_svd

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data("train")
vocab_size = len(word_to_id)
print("counting co-occurrence ...")
C = create_co_matrix(corpus, vocab_size, window_size)
print("calculating PPMI ...")
W = ppmi(C, verbose=True)

print("calculating SVD ...")
U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)

word_vecs = U[:, :wordvec_size]

querys = ["you", "year", "car", "toyota"]
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
