import sys

sys.path.append("..")
import numpy as np
from common_starter.util import (
    preprocess,
    create_co_matrix,
    cos_similarity,
    most_similar,
    ppmi,
)


text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)

C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

np.set_printoptions(precision=3)
print("convariance matrix")
print(C)
print("-" * 50)
print("PPMI")
print(W)
