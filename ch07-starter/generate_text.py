# coding: utf-8
import sys

sys.path.append("..")
from rnnlm_gen import BetterRnnlmGen
from dataset import ptb
import numpy as np

corpus, word_to_id, id_to_word = ptb.load_data("train")
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = BetterRnnlmGen()
model.load_params("../ch06-starter/BetterRnnlm.pkl")

start_word = "you"
start_id = word_to_id[start_word]
skips_words = ["N", "<unk>", "$"]
skip_ids = [word_to_id[w] for w in skips_words]

word_ids = model.generate(start_id, skip_ids)
txt = " ".join(id_to_word[i] for i in word_ids)
txt = txt.replace(" <eos>", ".\n")
print(txt)

model.reset_state()

start_words = "the meaning of life is"
start_ids = [word_to_id[w] for w in start_words.split(" ")]

for x in start_ids[:-1]:
    x = np.array(x).reshape(1, 1)
    model.predict(x)

word_ids = model.generate(start_ids[-1], skip_ids)
word_ids = start_ids[:-1] + word_ids
txt = " ".join([id_to_word[i] for i in word_ids])
txt = txt.replace(" <eos>", ".\n")
print("-" * 50)
print(txt)
