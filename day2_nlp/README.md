# Day2 自然言語処理

## 環境
- マシン: MacBook Air（Apple M1, メモリ 16 GB）
- OS: macOS 15.5
- Python: 3.12.16
- PyTorch: 2.9.1
- 計算デバイス: CPU（`gpu_id = None`）
- 依存パッケージ: リポジトリ直下の `requirements.txt` を参照


<!-- 演習を追加するときはこのブロックをコピーする
## 演習2.7.x タイトル
```sh
実行したコマンド
```
- 変更点: 変更したファイルと箇所（なければ「なし」）
- 結果: 数値や出力（図があればパスも）
-->

## 演習2.7.1 RNN言語モデルのテスト
```sh
python language_model_rnn_test.py
```
- 変更点: なし（10エポック目のモデルを使用）
- 結果: 

| 入力 | 出力（入力を含む） |
|---|---|
|私|私 は 泳ぐ の を 楽しんだ 。 |
|彼|彼 は 東洋 芸術 の かなり の 専門家 だ 。 |
|私 は ここ|私 は ここ で あれば よい のに 。 |
|今日 は|今日 は どうもありがとう 。|
|東京|東京 に 基づく の は 不可能 だ 。 |

## 演習2.7.2 LSTM言語モデルの学習
```sh
python language_model_lstm_train.py
```
- 変更点: language_model_lstm.py の forward() を実装
  - 単語を埋め込み、LSTMCell で状態を更新し、hidden state（self.hr[0]）から出力を計算
- 結果: mean loss（1文あたりの損失の合計の平均）

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean loss | 50.36 | 41.19 | 36.19 | 31.46 | 27.02 | 22.82 | 19.07 | 15.55 | 12.59 | 10.19 |

## 演習2.7.2 LSTM言語モデルのテスト
```sh
python language_model_lstm_test.py
```
- 変更点: なし（10エポック目のモデルを使用）
- 結果:

| 入力 | RNN の出力 | LSTM の出力 |
|---|---|---|
| 私 | 私 は 泳ぐ の を 楽しんだ 。 | 私 は 泳ぐ の を 楽しんだ 。 |
| 彼 | 彼 は 東洋 芸術 の かなり の 専門家 だ 。 | 彼 は 東洋 芸術 の かなり の 専門家 だ 。 |
| 私 は ここ | 私 は ここ で あれば よい のに 。 | 私 は ここ に カナダ 人 の メンバー で 成り立っている 。 |
| 今日 は | 今日 は どうもありがとう 。 | 今日 は 波 の なすがまま であった 。 |
| 東京 | 東京 に 基づく の は 不可能 だ 。 | 東京 に 行く 途中 で 財布 を 無くした 。 |

## 演習2.7.3 英語の言語モデルの学習
```sh
python language_model_lstm_train.py
```
- 変更点: language_model_lstm_train.py
  - 11行目: 語彙数を英語に変更
  - 22行目: 学習データを英語の文に変更
  - 48行目: 日本語モデルを上書きしないよう、保存名に `_en_` を追加
- 結果: mean loss（1文あたりの損失の合計の平均）

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean loss |51.58 | 43.08 | 37.83 | 32.94 | 28.37 | 24.22 | 20.63 | 17.41 | 14.70 | 12.45 | 

## 演習2.7.3 英語の言語モデルのテスト
```sh
python language_model_lstm_test.py
```
- 変更点: language_model_lstm_test.py
  - 11行目: 語彙数を英語に変更
  - 13行目: 読み込むモデルを `_en_` 付きに変更
  - 25, 44行目: 単語とIDの変換を英語用に変更
- 結果: 10エポック目のモデルを使用

| 入力 | 出力（入力を含む） |
|---|---|
|i| i have a slight fever . |
|he|he is a fine fix you got the cradle is very well .|
|she|she is a woman with a classical education . |
|i am|i am sure that jim made up that story . |
|this is|this is the main of people are still the world .|
|the|the thief ran away when she saw the policeman .|
|today|today 's tigers-giants baseball game was called off due to heavy rain . |
|tokyo|tokyo , i would n't pay tax at the station . |

## 演習2.7.4 英日翻訳モデルの学習
```sh
python translator_model_train.py
```
- 変更点: なし
- 結果: mean loss（1文あたりの損失の合計の平均）

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean loss | 55.45 | 46.69 | 41.60 | 36.71 | 32.06 | 27.66 | 23.48 | 19.86 | 16.60 | 13.81 |

- コードの対応箇所（translator_model.py）

| 探すもの | 行 | 内容 |
|---|---|---|
| ネットワークの定義 | 14-18 | 英語・日本語の埋め込み、エンコーダとデコーダの LSTM、出力層 |
| エンコード | 36-38, 51-52 | 英語の単語を順に LSTM に入力し、状態を更新する（単語の予測はしない） |
| エンコーダからデコーダへの受け渡し | 54-57 | h を引き継ぎ、c は 0 にして、EOS を最初の入力にする |
| デコード | 40-43 | LSTM の状態を更新し、次の単語のスコアを出力する |
| 学習時の処理 | 59-68 | 末尾に EOS を付けた正解との損失を合計し、正解の単語を次の入力にする |
| 推論時の処理 | 69-83 | 最も確率の高い単語を次の入力にし、EOS か 30 単語で止める |

- わかったこと
  - 英文全体は、文の長さによらず 100 次元の h に圧縮されてデコーダに渡される。
  - EOS は、デコーダの開始の合図（入力）と、文の終わりの判定（出力の正解）の2か所で使われる。

## 演習2.7.5 英日翻訳モデルのテスト
```sh
python translator_model_test.py
```
- 変更点: なし（10エポック目のモデルを使用）
- 結果:

| 入力 | 出力 | 学習データの正解 |
|---|---|---|
| i have a book . | 私はその食料のうちの少しを予備に取っておくべきだ。 | （学習データにない） |
| she is a teacher . | 彼女は古典の教育を受けている。 | （学習データにない） |
| i like tokyo . | 私はその食料のうちの少しを予備に取っておくべきだ。 | （学習データにない） |
| this is just what i wanted . | この静かな生活にはいい加減飽きてしまった。 | これが ちょうど 私が 欲しかった もの です 。 |
| thank you for today . | 人は波のなすがままであった。 | 今日は どうもありがとう 。 |
| it is in the kitchen . | 私達はみんなだれ一人として完璧な人間はいないことは分かっている。 | 台所に あります 。 |

観察
  - 日本語らしい文は出るが、翻訳としては誤っている。

## 演習2.7.6 translator_full.model による翻訳
```sh
python translator_model_test.py
```
- 変更点: translator_model_test.py
  - 7行目: 語彙の作成元を `data_full.txt` に変更
  - 12行目: 読み込むモデルを `translator_full.model` に変更
- 結果:

| 入力 | 出力 |
|---|---|
| i have a book . | 私はその本を読むべきだった。 |
| she is a teacher . | 彼女は先生ですか。 |
| i like tokyo . | 私は普通六歳だ。 |
| this is just what i wanted . | これは私が先日の日曜に通り過ぎた。 |
| thank you for today . | 今日はご親切にありがとうございます。 |
| it is in the kitchen . | それは名前です。 |
