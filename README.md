# 人工知能演習 基本課題

## 環境
- ローカル: macOS (arm64), Python 3.12, PyTorch 2.9.1（CPUで動作確認）
- 学習: Google Colab (GPU)
- 依存パッケージ: requirements.txt（Colab での実行時の一覧は各 result/packages.txt）

## 構成
| フォルダ | 内容 | 該当演習 |
|---|---|---|
| day2_nlp | RNN/LSTM言語モデル, 英日翻訳 | 2.7.x |
| day2_rl | 迷路のQ学習, DQN, CartPole | 3.4.x |
| day3_cv | MNIST MLP/CNN, CIFAR-10, 深層特徴検索 | 4.2.x, 4.3.x |
| day4_speech | 声質変換 | 5.4.x |

## データの準備
データは容量の都合でリポジトリに含めていない。
人工知能演習ウェブサイトの配布フォルダから、各 day フォルダの同じ位置に `data/` などを置く。

## 実行方法
各フォルダの README を参照。Colab では colab/run.ipynb を上から実行する。
