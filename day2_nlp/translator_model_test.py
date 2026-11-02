import argparse
import sys
import torch

import sentence_data
from translator_model import TranslatorModel

# 演習2.7.6: --model で使うモデルを切り替える（small: 演習2.7.5 の自分で学習したモデル，full: translator_full.model）
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["small", "full"], default="small")
args = parser.parse_args()

# 演習2.7.6: モデルは学習時と同じ語彙でしか使えないので，データとモデルを一緒に切り替える
if args.model == "full":
    data_file = "dataset/data_full.txt"
    model_file = "trained_model/translator_full.model"
else:
    data_file = "dataset/data_1000.txt"
    model_file = "trained_model/translator_10.model"

dataset = sentence_data.SentenceData(data_file)

model = TranslatorModel(dataset.english_word_size(),
                        dataset.japanese_word_size())

model.load_state_dict(torch.load(model_file))

# 入力された文章を単語に分割する
sentence = input("input an english sentence : ").split(' ')
# 単語IDのリストに変換する
sentence_id = []
for word in sentence:
    if not word:
        # 単語が空だったら飛ばす
        continue
    word = word.lower()
    id = dataset.english_word_id(word)
    if id is None:
        sys.stderr.write("Error : Unknown word " + word + "\n")
        sys.exit()
    else:
        id = torch.tensor(id,dtype=torch.long).unsqueeze(-1)
        sentence_id.append(id)

japanese = model(torch.stack(sentence_id))
for id in japanese:
    print(dataset.japanese_word(id), end='')
print()
