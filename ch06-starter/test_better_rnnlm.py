# coding: utf-8
import sys

sys.path.append("..")
from common import config

# GPUで実行する場合は下記のコメントアウトを消去（要cupy）
# ==============================================
# config.GPU = True
# ==============================================
from common.optimizer import SGD
from common.util import eval_perplexity
from dataset import ptb
from better_rnnlm import BetterRnnlm

# ハイパーパラメータの設定
batch_size = 20
wordvec_size = 650
hidden_size = 650
time_size = 35
lr = 20.0
max_epoch = 40
max_grad = 0.25
dropout = 0.5

# 学習データの読み込み
corpus, word_to_id, id_to_word = ptb.load_data("train")
corpus_val, _, _ = ptb.load_data("val")
corpus_test, _, _ = ptb.load_data("test")

vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

model = BetterRnnlm(vocab_size, wordvec_size, hidden_size, dropout)

# テストデータでの評価
model.load_params(file_name="BetterRnnlm.pkl")
ppl_test = eval_perplexity(model, corpus_test)
print("test perplexity: ", ppl_test)

