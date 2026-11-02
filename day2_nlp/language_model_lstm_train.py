import argparse
import sys
import torch
from torch import optim

import sentence_data
from sentence_data import EOS_ID
from language_model_lstm import LanguageModelLSTM

# 演習2.7.3: --lang で学習する言語を切り替える（ja: 配布時の動作，en: 英語）
parser = argparse.ArgumentParser()
parser.add_argument("--lang", choices=["ja", "en"], default="ja")
args = parser.parse_args()

dataset = sentence_data.SentenceData("dataset/data_1000.txt")

# 演習2.7.3: 言語に応じて語彙数，学習データ，保存名を切り替える
# en は日本語モデルを上書きしないよう，保存名に _en_ を付ける
if args.lang == "en":
    word_size = dataset.english_word_size()
    sentences = dataset.english_sentences()
    model_prefix = "trained_model/langage_model_lstm_en_"
else:
    word_size = dataset.japanese_word_size()
    sentences = dataset.japanese_sentences()
    model_prefix = "trained_model/langage_model_lstm_"

model = LanguageModelLSTM(word_size)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters())

epoch_num = 10
for epoch in range(epoch_num):
    print("{0} / {1} epoch start.".format(epoch + 1, epoch_num))

    sum_loss = 0.0
    for i, sentence in enumerate(sentences):
        model.reset_state()
        optimizer.zero_grad()
        accum_loss = None
        # 文の1単語目を入力して出力された2単語目，
        # 文の1単語目と2単語目を入力して出力された3単語目，のように，
        # 文の1～n-1単語目を入力して出力されたn単語目を全て確認して，
        # accum_lossに加算する
        # 最後の単語を入力した後，EOSが正しく出力されるかどうかも確認する
        for cur_word, next_word in zip(sentence, sentence[1:] + [EOS_ID]):
            cur_word = torch.tensor(cur_word,dtype=torch.long).unsqueeze(-1)
            next_word = torch.tensor(next_word,dtype=torch.long).unsqueeze(-1)
            out = model(cur_word)
            loss = loss_fn(out, next_word.to(out.device))
            accum_loss = loss if accum_loss is None else accum_loss + loss
        accum_loss.backward()
        optimizer.step()
        sum_loss += float(accum_loss.data.cpu())

        if (i + 1) % 100 == 0:
            print("{0} / {1} sentences finished.".format(
                i + 1, dataset.sentences_size()))

    print("mean loss = {0}.".format(sum_loss / dataset.sentences_size()))

    # 1 epoch 毎にファイルに書き出す
    model_file = model_prefix + str(epoch + 1) + ".model"
    torch.save(model.state_dict(), model_file)
