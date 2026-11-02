import argparse
import sys
import torch
from torch import optim

import sentence_data
from sentence_data import EOS_ID
from language_model_lstm import LanguageModelLSTM

# 演習2.7.3: --lang でテストする言語を切り替える（ja: 配布時の動作，en: 英語）
parser = argparse.ArgumentParser()
parser.add_argument("--lang", choices=["ja", "en"], default="ja")
args = parser.parse_args()

dataset = sentence_data.SentenceData("dataset/data_1000.txt")

# 演習2.7.3: 言語に応じて語彙数，読み込むモデル，単語とIDの変換を切り替える
if args.lang == "en":
    word_size = dataset.english_word_size()
    model_file = "trained_model/langage_model_lstm_en_10.model"
    word_to_id = dataset.english_word_id
    id_to_word = dataset.english_word
else:
    word_size = dataset.japanese_word_size()
    model_file = "trained_model/langage_model_lstm_10.model"
    word_to_id = dataset.japanese_word_id
    id_to_word = dataset.japanese_word

model = LanguageModelLSTM(word_size)

model.load_state_dict(torch.load(model_file))
model.reset_state()
model.eval()

initial_words = input("input initial words : ").split(' ')

next_y = None
for word in initial_words:
    if not word:
        # 単語が空だったら飛ばす
        continue
    print(word)
    word_id = word_to_id(word)
    if word_id is None:
        sys.stderr.write("Error : Unknown word " + word + "\n")
        sys.exit()
    # 単語をLSTMに入力する
    word_id = torch.tensor(word_id,dtype=torch.long).unsqueeze(-1)
    next_y = model(word_id)

if next_y is None:
    sys.stderr.write("Error : Empty input\n")
    sys.exit()

# 最大30単語で打ち切る
for i in range(30):
    # もっともそれらしい単語を取ってくる
    word_id = next_y.argmax().unsqueeze(-1)
    # EOSが出力されたら終了
    if word_id == EOS_ID:
        sys.exit()
    print(id_to_word(word_id))
    next_y = model(word_id)
